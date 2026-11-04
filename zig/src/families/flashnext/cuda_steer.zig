//! The GLP control vector on the CUDA device: each steered layer's unit direction and the launch
//! that subtracts it from the pass's residual streams after the layer's write-back (weightless,
//! in-model steering). The serve that opens a steered model holds the one plan; the pass reads it.

const std = @import("std");
const cuda = @import("cuda");
const Allocator = std.mem.Allocator;

/// How the vector acts: subtract the component along the direction (the CUDA lane's only mode).
pub const Mode = enum(u8) { project, add };

/// One layer's direction: the checkpoint's layer id and its unit vector, hidden floats wide.
pub const Layer = struct { id: u32, dir: []const f32 };

/// What the plan is built from: the vector's directions, the width and the stream count they fit.
pub const Source = struct {
    mode: Mode,
    alpha: f32,
    hidden: u32,
    /// The checkpoint's number of blocks: the plan covers ids below it.
    layers_total: u32,
    /// The hyper-connection streams a row carries (the pass's own streams count).
    streams: u32,
    layers: []const Layer,
};

pub const InitError = error{ OutOfMemory, BadSource, BuiltWithoutKernels, Cuda };

/// A control vector on the device: one direction buffer a layer, and the projection launch.
pub const Plan = struct {
    a: Allocator,
    alpha: f32,
    hidden: u32,
    streams: u32,
    /// Kept alive while the function below points into it.
    module: cuda.Module,
    project_fn: cuda.Function,
    /// The device copy of each steered layer's unit direction (null: the layer runs untouched).
    dirs: []?cuda.DeviceBuffer,

    /// Put the vector on the device; a direction whose width or layer id does not fit the model refuses.
    pub fn init(d: *const cuda.Driver, a: Allocator, src: Source) InitError!Plan {
        if (src.mode != .project) return error.BadSource; // the CUDA lane subtracts the refusal component only
        if (src.hidden == 0 or src.streams == 0) return error.BadSource;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var plan: Plan = .{
            .a = a,
            .alpha = src.alpha,
            .hidden = src.hidden,
            .streams = src.streams,
            .module = undefined,
            .project_fn = undefined,
            .dirs = try a.alloc(?cuda.DeviceBuffer, @max(src.layers_total, 1)),
        };
        for (plan.dirs) |*slot| slot.* = null;
        errdefer {
            for (plan.dirs) |*slot| if (slot.*) |*b| b.free();
            a.free(plan.dirs);
        }
        for (src.layers) |l| {
            if (l.id >= src.layers_total or l.dir.len != src.hidden) return error.BadSource;
            if (plan.dirs[l.id] != null) return error.BadSource; // a layer steers once
            const bytes = std.mem.sliceAsBytes(l.dir);
            plan.dirs[l.id] = cuda.DeviceBuffer.fromHost(d, bytes) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.Cuda,
            };
        }
        plan.module = cuda.Module.load(d, cuda.kernels.flashnext_glp) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Cuda,
        };
        plan.project_fn = plan.module.function("fn_glp_project") catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Cuda,
        };
        return plan;
    }

    pub fn deinit(self: *Plan) void {
        for (self.dirs) |*slot| if (slot.*) |*b| b.free();
        self.module.unload();
        self.a.free(self.dirs);
    }

    /// Whether any layer steers; a plan whose every slot is null is the same as none.
    pub fn covers(self: *const Plan) bool {
        for (self.dirs) |slot| if (slot != null) return true;
        return false;
    }

    /// Subtract this layer's direction from every row's streams, in place, after the write-back.
    /// The pass calls it for each layer it runs; a layer the vector does not cover launches nothing.
    pub fn projectLayer(self: *const Plan, stream: cuda.Stream, h: u64, rows: usize, layer: usize) !void {
        if (rows == 0) return;
        const b = self.dirs[layer] orelse return;
        var args: cuda.Args = .{};
        args.add(b.ptr);
        args.add(h);
        args.add(self.alpha);
        args.add(@as(c_int, @intCast(self.hidden)));
        args.add(@as(c_int, @intCast(self.streams)));
        try cuda.launch.launch(self.project_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
    }
};

/// The plan the process serves with: one model, one vector, set once at open and read by every pass.
var current: ?*Plan = null;

pub fn set(p: ?*Plan) void {
    current = p;
}

pub fn get() ?*Plan {
    return current;
}
