//! The GLP control vector on the CUDA device: each steered layer's unit direction and the launch
//! that subtracts it from the pass's residual streams after the layer's write-back (weightless,
//! in-model steering). The serve that opens a steered model holds the one plan; the pass reads it.

const std = @import("std");
const cuda = @import("cuda");
const steer_source = @import("steer_source.zig");
const Allocator = std.mem.Allocator;

/// How the vector acts: subtract the component along the direction (the CUDA lane's only mode).
pub const Mode = enum(u8) { project, add };

/// One layer's direction: the checkpoint's layer id and its f32 row over the materialized stream.
pub const Layer = steer_source.Layer;

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
    /// A failed upload frees the buffers that had landed, so a retry of the same vector is safe:
    /// no device or host memory is held behind a refusal.
    pub fn init(d: *const cuda.Driver, a: Allocator, src: Source) InitError!Plan {
        if (src.mode != .project) return error.BadSource; // the CUDA lane subtracts the refusal component only
        if (src.hidden == 0 or src.streams == 0) return error.BadSource;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        for (src.layers) |l| {
            if (l.id >= src.layers_total) return error.BadSource;
        }
        // The shape rules and unit normalization live in the device-free steer_source, so they are tested.
        const dirs = steer_source.prepared(a, src.hidden, src.streams, src.layers_total, src.layers) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.BadSource => return error.BadSource,
        };
        errdefer { // the array's errdefer is installed last: it can never double-free with the uploads' below
            for (dirs) |dd| if (dd) |v| a.free(v);
            a.free(dirs);
        }
        var module = cuda.Module.load(d, cuda.kernels.flashnext_glp) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Cuda,
        };
        errdefer module.unload(); // ours: a refused vector leaves no module loaded on the device
        const project_fn = module.function("fn_glp_project") catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Cuda,
        };
        // The slots are filled by one all-or-nothing block: a buffer that failed to reach the device
        // frees the ones already uploaded, so the array's own cleanup never sees a freed buffer.
        const slots = blk: {
            const s = a.alloc(?cuda.DeviceBuffer, @max(src.layers_total, 1)) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => unreachable,
            };
            var uploaded: usize = 0;
            errdefer {
                for (s[0..uploaded]) |*slot| if (slot.*) |*b| b.free();
                a.free(s);
            }
            for (s) |*slot| slot.* = null;
            for (dirs, 0..) |dd, i| {
                const dir = dd orelse continue;
                s[i] = cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(dir)) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => return error.Cuda,
                };
                uploaded = i + 1;
            }
            break :blk s;
        };
        return .{ .a = a, .alpha = src.alpha, .hidden = src.hidden, .streams = src.streams, .module = module, .project_fn = project_fn, .dirs = slots };
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
