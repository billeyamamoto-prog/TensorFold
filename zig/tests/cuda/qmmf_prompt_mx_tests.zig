//! The relaxed MXFP8 prompt tile (qmmf.cu prompt_mx_body, TF_FLASHNEXT_PROMPT_MX128) at Flash Next's MXFP8 shapes: FP64-bounded,
//! each row's bytes the same whatever rows share its call or the tile it falls in, bf16 out the rounded fp32 sum, then timed
//! against the exact lane.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Shape = struct { name: []const u8, n: usize, k: usize };

/// The MXFP8 projections a Flash Next prompt pass puts on the tile (1,024 columns or more), and an edge of a partial tile.
const shapes = [_]Shape{
    .{ .name = "DeltaNet input", .n = 16480, .k = 2560 },
    .{ .name = "attention input", .n = 13952, .k = 2560 },
    .{ .name = "mixer output", .n = 2560, .k = 6144 },
    .{ .name = "partial tile", .n = 1088, .k = 640 },
};
const full_rows: usize = 3072;
const cuts = [_]usize{ 1, 77, 128, 300 };
const shift: usize = 5;

/// A random MXFP8 weight packed for the lane, and its exact float64 values [n][k].
fn build(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, sh: Shape) !struct { w: cuda.qmmf.Weight, deq: []f64 } {
    const codes = try a.alloc(u8, sh.n * sh.k);
    for (codes) |*c| c.* = rng.int(u8) & 0xF7; // no e4m3 NaN (0x7F, 0xFF)
    const scales = try a.alloc(u8, sh.n * sh.k / 32);
    for (scales) |*s| s.* = 120 + rng.uintLessThan(u8, 8);
    const deq = try a.alloc(f64, sh.n * sh.k);
    for (deq, 0..) |*o, i| o.* = @import("core").quant.fp8.e4m3(codes[i]) * std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, scales[i / 32]) - 127)));
    const pc = try a.alloc(u8, cuda.mxfp8.padded(sh.n) * sh.k);
    cuda.mxfp8.packCodes(pc, codes, sh.n, sh.k);
    const ps = try a.alloc(u8, cuda.mxfp8.padded(sh.n) * (sh.k / 32));
    cuda.mxfp8.packScales(ps, scales, sh.n, sh.k);
    return .{ .w = cuda.mxfp8.weight(try r.dev(pc), try r.dev(ps), sh.n, sh.k), .deq = deq };
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x3128_0a);
    var r = try rig.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    try lane.loadPromptMx(1024);
    for (shapes) |sh| {
        const built = try build(&r, a, prng.random(), sh);
        const w = built.w;
        try check.expect(lane.takesPromptMx(w), "{s}: the tile takes an MXFP8 weight of {d} columns", .{ sh.name, sh.n });
        const ldx = sh.k + 64; // rows apart, as a pass's views are
        const xs = r.bfs((full_rows + shift) * ldx, 1.0);
        const x = try r.dev(std.mem.sliceAsBytes(xs));
        const f32_bytes = full_rows * sh.n * 4;
        const full = try r.out(f32_bytes);
        try lane.matmulPromptMx(r.s, x, ldx, full_rows, w, full, .f32);
        try r.s.synchronize();
        const want = try r.back(f32, full, full_rows * sh.n);
        // fewer rows, and rows shifted off the 128-row tiles: each row's bytes the full call's
        for (cuts) |m| {
            const y = try r.out(m * sh.n * 4);
            try lane.matmulPromptMx(r.s, x, if (m == 1) sh.k else ldx, m, w, y, .f32);
            try r.s.synchronize();
            try check.sameBytes("prompt tile rows at fewer rows", std.mem.sliceAsBytes(try r.back(f32, y, m * sh.n)), std.mem.sliceAsBytes(want[0 .. m * sh.n]));
        }
        // the 64 x 64 form on many rows, the 128 x 128 one on a few: each row's bytes the same
        for ([_]usize{ 300, 40 }, [_]bool{ true, false }) |m, small| {
            const y = try r.out(m * sh.n * 4);
            try lane.matmulPromptMxTile(r.s, x, ldx, m, w, y, .f32, small);
            try r.s.synchronize();
            try check.sameBytes("prompt tile forms' rows", std.mem.sliceAsBytes(try r.back(f32, y, m * sh.n)), std.mem.sliceAsBytes(want[0 .. m * sh.n]));
        }
        const moved = try r.out((full_rows - shift) * sh.n * 4);
        try lane.matmulPromptMx(r.s, x + shift * ldx * 2, ldx, full_rows - shift, w, moved, .f32);
        try r.s.synchronize();
        try check.sameBytes("prompt tile rows on other tiles", std.mem.sliceAsBytes(try r.back(f32, moved, (full_rows - shift) * sh.n)), std.mem.sliceAsBytes(want[shift * sh.n ..]));
        // bf16 out: the fp32 sum rounded once
        const yb = try r.out(full_rows * sh.n * 2);
        try lane.matmulPromptMx(r.s, x, ldx, full_rows, w, yb, .bf16);
        try r.s.synchronize();
        const got_b = try r.back(u16, yb, full_rows * sh.n);
        for (got_b, want) |g, f| try check.expect(g == rig.bf16(f), "{s}: bf16 out is the fp32 sum rounded ({x} vs {x})", .{ sh.name, g, rig.bf16(f) });
        // FP64 bound on sampled rows, and against the exact lane's fp32 rows
        const part = try r.out(@max(16, lane.partBytes(full_rows, w)));
        const exact = try r.out(f32_bytes);
        try lane.matmulOut(r.s, x, ldx, full_rows, w, exact, .f32, part);
        try r.s.synchronize();
        const lane_rows = try r.back(f32, exact, full_rows * sh.n);
        var worst: f64 = 0;
        for ([_]usize{ 0, 1, 129, 1500, full_rows - 1 }) |row| for (0..sh.n) |j| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (0..sh.k) |i| {
                const t = rig.bf(xs[row * ldx + i]) * built.deq[j * sh.k + i];
                sum += t;
                mag += @abs(t);
            }
            const tol = rig.sumTol(mag, sh.k) + 1e-6;
            try r.within("prompt tile fp32 vs FP64", want[row * sh.n + j], sum, tol);
            try r.within("exact lane fp32 vs FP64", lane_rows[row * sh.n + j], sum, tol);
            worst = @max(worst, @abs(@as(f64, want[row * sh.n + j]) - lane_rows[row * sh.n + j]) / tol);
        };
        // time both on the chunk's rows
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            const reps = 20;
            const t0 = check.now(gpu.io);
            for (0..reps) |_| {
                if (which == 0) try lane.matmulOut(r.s, x, ldx, full_rows, w, exact, .f32, part) else try lane.matmulPromptMx(r.s, x, ldx, full_rows, w, full, .f32);
            }
            try r.s.synchronize();
            us[which] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / reps;
        }
        // short calls (a turn's new rows): the exact lane, the 128 x 128 tile and its 64 x 64 form
        for ([_]usize{ 16, 40, 64, 128 }) |m| {
            var t: [3]f64 = undefined;
            for (0..3) |which| {
                const reps = 200;
                const t0 = check.now(gpu.io);
                for (0..reps) |_| switch (which) {
                    0 => try lane.matmulOut(r.s, x, ldx, m, w, exact, .f32, part),
                    1 => try lane.matmulPromptMxTile(r.s, x, ldx, m, w, full, .f32, false),
                    else => try lane.matmulPromptMxTile(r.s, x, ldx, m, w, full, .f32, true),
                };
                try r.s.synchronize();
                t[which] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / reps;
            }
            std.debug.print("qmmf-prompt-mx {s} m={d}: exact lane {d:.1} us, 128 tile {d:.1} us, 64 tile {d:.1} us\n", .{ sh.name, m, t[0], t[1], t[2] });
        }
        const flops = 2.0 * @as(f64, @floatFromInt(full_rows * sh.n * sh.k));
        std.debug.print("qmmf-prompt-mx {s} m={d} n={d} k={d}: exact lane {d:.0} us ({d:.1} TF), prompt tile {d:.0} us ({d:.1} TF), worst |tile - lane| {d:.3} of the bound\n", .{ sh.name, full_rows, sh.n, sh.k, us[0], flops / us[0] / 1e6, us[1], flops / us[1] / 1e6, worst });
    }
    r.report("relaxed MXFP8 prompt tile: FP64-bounded, row bytes independent of the call's rows and tiles, bf16 the rounded fp32");
}
