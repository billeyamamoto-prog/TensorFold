//! The vector's host-side shape rules, kept free of CUDA so the serving checks are testable
//! without a device: the direction spans the whole materialized hyper-connection stream
//! (hidden x streams — the width the shipped vectors are fit at), a layer steers once, and
//! every direction reaches unit length in f32 before it is uploaded.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Layer = struct { id: u32, dir: []const f32 };

pub const Error = error{ OutOfMemory, BadSource };

/// Validate the directions and return them layer-indexed, each a unit f32 row the whole
/// materialized stream spans (null: the vector is silent on that layer).
pub fn prepared(a: Allocator, hidden: u32, streams: u32, layers_total: u32, layers: []const Layer) Error![]?[]const f32 {
    if (hidden == 0 or streams == 0) return error.BadSource;
    const out = try a.alloc(?[]const f32, @max(layers_total, 1));
    errdefer {
        for (out) |d| if (d) |v| a.free(v);
        a.free(out);
    }
    @memset(out, null);
    for (layers) |l| {
        if (l.id >= layers_total or l.dir.len != @as(usize, hidden) * streams) return error.BadSource;
        if (out[l.id] != null) return error.BadSource; // a layer steers once
        const dir = try a.alloc(f32, l.dir.len);
        errdefer a.free(dir);
        @memcpy(dir, l.dir);
        var norm: f64 = 0;
        for (dir) |v| norm += @as(f64, v) * @as(f64, v);
        if (!(norm > 0) or !std.math.isFinite(norm)) return error.BadSource; // zero or NaN steers nothing and divides by nothing
        const inv: f32 = @floatCast(1.0 / @sqrt(norm));
        for (dir) |*v| v.* *= inv;
        out[l.id] = dir;
    }
    return out;
}

// ── unit tests ──────────────────────────────────────────────────────────────

test "the vector's row spans the whole materialized stream" {
    const a = std.testing.allocator;
    var dir: [8]f32 = undefined;
    for (&dir, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) + 1;
    const layers = [_]Layer{.{ .id = 2, .dir = &dir }};
    const got = try prepared(a, 2, 4, 6, &layers); // hidden x streams = 8
    defer a.free(got);
    defer if (got[2]) |v| a.free(v);
    try std.testing.expect(got[0] == null and got[5] == null);
    const d = got[2].?;
    try std.testing.expectEqual(@as(usize, 8), d.len);
    var norm: f64 = 0;
    for (d) |v| norm += @as(f64, v) * @as(f64, v);
    try std.testing.expectApproxEqAbs(@as(f64, 1), norm, 1e-5); // unit, however the file brought it
}

test "a too-narrow direction, a second layer steer and a zero direction all refuse" {
    const a = std.testing.allocator;
    const narrow = [_]f32{ 1, 0, 0, 0 }; // hidden-width, not the whole stream
    const wide = [_]f32{ 1, 1, 1, 1, 1, 1, 1, 1 };
    const zero = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expectError(error.BadSource, prepared(a, 2, 4, 6, &.{.{ .id = 0, .dir = &narrow }}));
    try std.testing.expectError(error.BadSource, prepared(a, 2, 4, 6, &.{ .{ .id = 1, .dir = &wide }, .{ .id = 1, .dir = &wide } }));
    try std.testing.expectError(error.BadSource, prepared(a, 2, 4, 6, &.{.{ .id = 1, .dir = &zero }}));
    try std.testing.expectError(error.BadSource, prepared(a, 2, 4, 6, &.{.{ .id = 6, .dir = &wide }})); // no layer 6 of a 6-layer model
}
