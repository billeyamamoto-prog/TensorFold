//! Generate a GLP control vector file with the engine's own writer (test and A/B use).
const std = @import("std");
const glp = @import("glp.zig");

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const out_path = args[1];
    const seed: u64 = try std.fmt.parseInt(u64, args[2], 10);
    const alpha: f32 = if (args.len > 3) try std.fmt.parseFloat(f32, args[3]) else 1.0;

    var r = std.Random.DefaultPrng.init(seed);
    const rnd = r.random();
    var layers = std.ArrayList(glp.Layer).empty;
    var id: u32 = 1;
    while (id <= 47) : (id += 1) {
        const dir = try a.alloc(f32, 2560);
        for (dir) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
        var n: f64 = 0;
        for (dir) |v| n += @as(f64, v) * @as(f64, v);
        const inv: f32 = @floatCast(1.0 / @sqrt(n));
        for (dir) |*v| v.* = v.* * inv;
        try layers.append(a, .{ .id = id, .dir = dir });
    }
    const bytes = try glp.write(a, .{
        .mode = .project,
        .alpha = alpha,
        .base_model = "Qwen/Qwen3.8-Flash-Next",
        .method = "synthetic random unit directions (engine writer; A/B and startup checks)",
        .layers = layers.items,
    });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = bytes });
    std.debug.print("wrote {s}: {d} bytes, {d} layers, alpha {d}\n", .{ out_path, bytes.len, layers.items.len, alpha });
}
