//! Flash Next prompt states for the shared prompt cache: one stream's caches after `at` tokens, copied on the device.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;
const pages = @import("cuda_pages.zig");
const sizes = @import("cuda_seq_bytes.zig");
const Heads = @import("cuda_heads.zig").Heads;
const Kv = @import("cuda_kv.zig").Kv;

const index_row = wt.index_dim * 2;
const tail_bytes = wt.ple_tail_n * wt.wide * 2;
/// The MTP head's cache: its own layer, whole on every rank.
const head_heads: Heads = .full;

/// A state's shape for its disk image (cuda_disk.zig, one rank, the whole layer): layer counts and bytes a row.
pub const shape = .{ .linear = wt.linear_n, .attn = wt.attn_n, .conv = sizes.convBytes(.full), .state = sizes.stateBytes(.full), .kv_row = Heads.full.kvRow(), .index_row = index_row, .ratio = wt.index_ratio, .tail = tail_bytes };

/// A key row and a value row of one position at heads `h` in format `kv` (--kv-cache fp8: whole heads, cuda_kv.zig).
fn keyRow(h: Heads, kv: Kv) usize {
    return if (kv == .fp8) kv.keyRow() else h.kvRow();
}
fn valueRow(h: Heads, kv: Kv) usize {
    return if (kv == .fp8) kv.valueRow() else h.kvRow();
}

/// One attention cache's first `at` key, value and indexer-key rows at heads `h`, and the pooled blocks they complete.
fn attnBytes(at: usize, h: Heads, kv: Kv) usize {
    return at * (keyRow(h, kv) + valueRow(h, kv) + index_row) + at / wt.index_ratio * index_row;
}

/// Device bytes a state at the model's heads `h` after `at` tokens takes: fixed DeltaNet pieces, attention by token.
pub fn bytes(at: usize, head: bool, h: Heads) usize {
    return bytesKv(at, h, .bf16, if (head) .bf16 else null);
}

/// bytes with the attention caches in format `kv` and the head's (when `head_kv` is not null) in its own: an fp8 state
/// takes 1,056 bytes of keys and values a layer and token where a bf16 one takes 2,048.
pub fn bytesKv(at: usize, h: Heads, kv: Kv, head_kv: ?Kv) usize {
    const own = wt.linear_n * (sizes.convBytes(h) + sizes.stateBytes(h)) + wt.attn_n * attnBytes(at, h, kv) + tail_bytes;
    return own + if (head_kv) |hk| attnBytes(at, head_heads, hk) else 0;
}

/// One kept state; `head`: it holds the MTP head's cache over the same rows, which a drafting stream needs.
pub const Snap = struct {
    rank_id: u64 = 0,
    buf: cuda.DeviceBuffer,
    at: usize,
    history: [2]i64,
    head: bool,
    /// The model's heads its pieces hold (the saved sequence's): a sequence of other heads cannot take it.
    heads: Heads = .full,
    /// The format of the attention rows it holds (the engine's --kv-cache).
    kv: Kv = .bf16,
    /// The format of the head's rows when it holds them (cuda_mtp_draft.zig headFormat).
    head_kv: Kv = .bf16,
};

/// Where a kept state's pieces sit in its buffer (move's order) for a pass's marks (Hooks.marks): whole heads only.
pub const layout = struct {
    const conv_bytes = sizes.convBytes(.full);
    const state_bytes = sizes.stateBytes(.full);
    /// A linear layer's conv window and the n-gram tail, in bytes.
    pub const conv_size = conv_bytes;
    pub const tail_size = tail_bytes;
    /// Linear layer li's conv window.
    pub fn conv(li: usize) u64 {
        return li * conv_bytes;
    }
    /// Linear layer li's DeltaNet state.
    pub fn state(li: usize) u64 {
        return wt.linear_n * conv_bytes + li * state_bytes;
    }
    /// The n-gram tail of a state after `n` tokens (after every attention cache's rows in format `kv`).
    pub fn tail(n: usize, kv: Kv) u64 {
        return wt.linear_n * (conv_bytes + state_bytes) + wt.attn_n * attnBytes(n, .full, kv);
    }
};

const Way = enum { save, restore, rows };

/// A piece `move` copies: `layers` buffers of `n` bytes, rows of `row` bytes (0: whole) a `per`-position block each.
pub const Piece = struct { layers: usize, n: usize, row: usize = 0, per: usize = 1 };

/// `move`'s pieces in order at `at` and the model's heads `h` (the disk image's at full): DeltaNet, attention, tail.
pub const Order = struct { p: [11]Piece, len: usize };

pub fn order(at: usize, head: bool, h: Heads) Order {
    return orderKv(at, h, .bf16, if (head) .bf16 else null);
}

/// order with the attention rows in format `kv` and the head's (when `head_kv` is not null) in its own.
pub fn orderKv(at: usize, h: Heads, kv: Kv, head_kv: ?Kv) Order {
    const head = head_kv != null;
    var o: [11]Piece = undefined;
    o[0] = .{ .layers = wt.linear_n, .n = sizes.convBytes(h) };
    o[1] = .{ .layers = wt.linear_n, .n = sizes.stateBytes(h) };
    o[2..6].* = attnOrder(wt.attn_n, at, h, kv);
    o[6] = .{ .layers = 1, .n = tail_bytes };
    if (head_kv) |hk| o[7..11].* = attnOrder(1, at, head_heads, hk);
    return .{ .p = o, .len = if (head) 11 else 7 };
}

/// An attention cache's pieces for `at` rows: complete pooled blocks only (the rest pool again as later rows land).
fn attnOrder(layers: usize, at: usize, h: Heads, kv: Kv) [4]Piece {
    const kr = keyRow(h, kv);
    const vr = valueRow(h, kv);
    return .{
        .{ .layers = layers, .n = at * kr, .row = kr },
        .{ .layers = layers, .n = at * vr, .row = vr },
        .{ .layers = layers, .n = at * index_row, .row = index_row },
        .{ .layers = layers, .n = at / wt.index_ratio * index_row, .row = index_row, .per = wt.index_ratio },
    };
}

/// Each piece of the sequence (and the head's cache) to or from the buffer at `base`; .rows: the attention rows only.
fn move(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, base: u64, at: usize, way: Way) !void {
    const dr = stream.d;
    var off: u64 = 0;
    // fp8 rows are contiguous only (cuda_prompt.zig newSeq refuses pages with them)
    if (seq.kv == .fp8 and seq.paged != null) return error.KvCachePaged;
    const head_kv: ?Kv = if (d) |h| h.kv else null;
    const o = orderKv(at, seq.heads, seq.kv, head_kv);
    var bufs: [11][]cuda.DeviceBuffer = undefined;
    bufs[0..7].* = .{ seq.conv, seq.state, seq.kc, seq.vc, seq.ikc, seq.pooled, (&seq.tail)[0..1] };
    if (d) |h| bufs[7..11].* = .{ (&h.kc)[0..1], (&h.vc)[0..1], (&h.ikc)[0..1], (&h.pooled)[0..1] };
    for (o.p[0..o.len], bufs[0..o.len], 0..) |p, layers, pi| {
        if (layers.len != p.layers) return error.SnapShape;
        for (layers, 0..) |b, layer| {
            if (p.n == 0) continue;
            if (way == .rows and (pi < 2 or pi == 6)) {
                off += p.n;
                continue;
            }
            // the model's caches on the stream's pages; the head's too in a pool lane (the layer after the model's)
            const head = pi >= 7 and d != null and d.?.pg.pt != 0;
            if ((pi >= 2 and pi < 6) or head) if (seq.paged) |*pg| {
                // the same bytes in the same order as a contiguous cache's, a run a page
                const part: pages.Part = @fromBackingInt(@intCast(if (head) pi - 7 else pi - 2));
                const at_layer = if (head) wt.attn_n else layer;
                const row = pages.rowBytes(part);
                if (row != p.row) return error.SnapShape;
                const per = pages.rows(part, pg.tokens);
                const n = p.n / row;
                var r: usize = 0;
                while (r < n) {
                    const run = @min(per - r % per, n - r);
                    const at_row = pg.rowAt(at_layer, part, r);
                    const dst, const src = if (way != .restore) .{ base + off, at_row } else .{ at_row, base + off };
                    try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, run * row, stream.handle), "cuMemcpyDtoDAsync");
                    off += run * row;
                    r += run;
                }
                continue;
            };
            const dst, const src = if (way != .restore) .{ base + off, b.ptr } else .{ b.ptr, base + off };
            try dr.check(dr.api.cuMemcpyDtoDAsync_v2(dst, src, p.n, stream.handle), "cuMemcpyDtoDAsync");
            off += p.n;
        }
    }
    std.debug.assert(off == bytesKv(at, seq.heads, seq.kv, head_kv));
}

/// A copy of `seq` (and of the head's cache, with `d`) as it stands after `seq.fed` tokens.
pub fn save(gpa: std.mem.Allocator, driver: *const cuda.Driver, stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft) !*Snap {
    const at = seq.fed;
    if (d) |h| if (h.len != at or h.drafted != 0) return error.NotAtMark;
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    const head_kv: ?Kv = if (d) |h| h.kv else null;
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytesKv(at, seq.heads, seq.kv, head_kv)), .at = at, .history = seq.history, .head = d != null, .heads = seq.heads, .kv = seq.kv, .head_kv = head_kv orelse .bf16 };
    errdefer snap.buf.free();
    errdefer cuda.completion.drain(driver, stream, "a failed snapshot save did not drain");
    try move(stream, seq, d, snap.buf.ptr, at, .save);
    try stream.synchronize();
    return snap;
}

/// A kept state for position `at` that a pass running past it fills (Hooks.marks); `finish` adds the attention rows.
pub fn prepare(gpa: std.mem.Allocator, driver: *const cuda.Driver, at: usize, history: [2]i64, heads: Heads, kv: Kv, head_kv: ?Kv) !*Snap {
    const snap = try gpa.create(Snap);
    errdefer gpa.destroy(snap);
    snap.* = .{ .buf = try cuda.DeviceBuffer.alloc(driver, bytesKv(at, heads, kv, head_kv)), .at = at, .history = history, .head = head_kv != null, .heads = heads, .kv = kv, .head_kv = head_kv orelse .bf16 };
    return snap;
}

/// After the pass ran past `snap.at`: the attention caches' first rows (and the head's) into `snap`, queued.
pub fn finish(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *Snap) !void {
    if (seq.fed < snap.at) return error.NotAtMark;
    if (snap.kv != seq.kv) return error.KvFormat;
    if (d) |h| if (!snap.head or h.len < snap.at or h.kv != snap.head_kv) return error.NotAtMark;
    try move(stream, seq, d, snap.buf.ptr, snap.at, .rows);
}

/// `seq` (and the head's cache, with `d`) becomes `snap`'s state; refused before any copy when it cannot hold it.
pub fn restore(stream: cuda.Stream, seq: *prompt.Seq, d: ?*Draft, snap: *const Snap) !void {
    if (snap.at > seq.cap) return error.SequenceTooShort;
    if (!snap.heads.eql(seq.heads)) return error.SnapShape;
    if (snap.kv != seq.kv) return error.KvFormat;
    if (d) |h| if (!snap.head or snap.at > h.cap or h.kv != snap.head_kv) return error.NoHeadState;
    // a state with the head's rows serves a stream without drafts too: those rows stay in the buffer
    try move(stream, seq, d, snap.buf.ptr, snap.at, .restore);
    try stream.synchronize();
    seq.history = snap.history;
    seq.fed = snap.at;
    if (d) |h| {
        h.reset();
        h.len = snap.at;
    }
}

pub fn drop(gpa: std.mem.Allocator, stream: cuda.Stream, snap: *Snap) void {
    cuda.completion.drain(stream.d, stream, "a snapshot drop did not drain");
    snap.buf.free();
    gpa.destroy(snap);
}

test "a state's bytes: about 110 MiB of DeltaNet states and the n-gram tail, then 27 KiB a token" {
    try std.testing.expectEqual(@as(usize, 12 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, false, .full) - bytes(0, false, .full));
    // a complete block of 4 adds its pooled row
    try std.testing.expectEqual(@as(usize, 12 * (4 * (2 * 2 * 256 * 2 + 128 * 2) + 128 * 2)), bytes(4, false, .full) - bytes(0, false, .full));
    // the head's cache is one more attention layer's rows
    try std.testing.expectEqual(@as(usize, 13 * (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, .full) - bytes(0, true, .full));
    try std.testing.expect(bytes(0, false, .full) > 100 << 20 and bytes(0, false, .full) < 120 << 20);
}

test "a state at a rank's half of the heads: half the conv windows, states and KV rows, the indexer and the head's whole" {
    const h = Heads.half;
    try std.testing.expectEqual(@as(usize, 36 * (3 * 5120 * 2 + 24 * 128 * 128 * 4) + 9 * 4 * 2560 * 2), bytes(0, false, h));
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2)), bytes(1, false, h) - bytes(0, false, h));
    // the head's cache keeps its two KV heads beside the model's one
    try std.testing.expectEqual(@as(usize, 12 * (2 * 256 * 2 + 128 * 2) + (2 * 2 * 256 * 2 + 128 * 2)), bytes(1, true, h) - bytes(0, true, h));
    const o = order(8, true, h);
    try std.testing.expectEqual(@as(usize, 3 * 5120 * 2), o.p[0].n);
    try std.testing.expectEqual(@as(usize, 24 * 128 * 128 * 4), o.p[1].n);
    for (o.p[2..4]) |p| try std.testing.expectEqual(@as(usize, 512), p.row);
    for (o.p[7..9]) |p| try std.testing.expectEqual(@as(usize, 1024), p.row);
    var sum: usize = 0;
    for (o.p[0..o.len]) |p| sum += p.layers * p.n;
    try std.testing.expectEqual(bytes(8, true, h), sum);
}

test "a state of the whole layer restores into no sequence of half the heads, and the reverse" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_][2]Heads{ .{ .full, .half }, .{ .half, .full } }) |pair| {
        const kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = 1, .history = .{ 0, 0 }, .head = false, .heads = pair[0] };
        var seq: prompt.Seq = .{ .heads = pair[1], .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = .{ .d = &driver, .ptr = 0, .len = 0 }, .history = .{ 0, 0 }, .fed = 0, .logits = &.{}, .cap = 8 };
        // refused before any copy: no driver call is made
        try std.testing.expectError(error.SnapShape, restore(stream, &seq, null, &kept));
        try std.testing.expectEqual(@as(usize, 0), seq.fed);
    }
}

test {
    _ = @import("cuda_paging_snap_test.zig");
}

test "an fp8 state takes 1,056 + 256 bytes a layer and token, its head's cache too, and its pieces add up" {
    try std.testing.expectEqual(@as(usize, 12 * (1056 + 256)), bytesKv(1, .full, .fp8, null) - bytesKv(0, .full, .fp8, null));
    try std.testing.expectEqual(@as(usize, 13 * (1056 + 256)), bytesKv(1, .full, .fp8, .fp8) - bytesKv(0, .full, .fp8, .fp8));
    for ([_]usize{ 0, 1, 4, 4096 }) |at| try std.testing.expectEqual(bytes(at, true, .full), bytesKv(at, .full, .bf16, .bf16));
    // a bf16 head under fp8 caches: 2,048 + 256 bytes a token for the head and 1,056 + 256 for each layer
    for ([_]usize{ 0, 1, 4, 513, 262144 }) |at| try std.testing.expectEqual(at * (2048 - 1056), bytesKv(at, .full, .fp8, .bf16) - bytesKv(at, .full, .fp8, .fp8));
    const o = orderKv(8, .full, .fp8, .fp8);
    try std.testing.expectEqual(@as(usize, 2 * 272), o.p[2].row);
    try std.testing.expectEqual(@as(usize, 2 * 256), o.p[3].row);
    var sum: usize = 0;
    for (o.p[0..o.len]) |p| sum += p.layers * p.n;
    try std.testing.expectEqual(bytesKv(8, .full, .fp8, .fp8), sum);
    const ob = orderKv(8, .full, .fp8, .bf16);
    try std.testing.expectEqual(@as(usize, 2 * 512), ob.p[7].row);
    // the in-pass marks' n-gram tail sits after the fp8 rows
    try std.testing.expectEqual(@as(u64, bytesKv(8, .full, .fp8, null) - tail_bytes), layout.tail(8, .fp8));
}

test "a state of one cache format restores into no sequence of the other" {
    var driver: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &driver, .handle = null };
    for ([_][2]Kv{ .{ .bf16, .fp8 }, .{ .fp8, .bf16 } }) |pair| {
        const kept: Snap = .{ .buf = .{ .d = &driver, .ptr = 0x100, .len = 1, .borrowed = true }, .at = 1, .history = .{ 0, 0 }, .head = false, .kv = pair[0] };
        var seq: prompt.Seq = .{ .heads = .full, .kv = pair[1], .conv = &.{}, .state = &.{}, .kc = &.{}, .vc = &.{}, .ikc = &.{}, .pooled = &.{}, .tail = .{ .d = &driver, .ptr = 0, .len = 0 }, .history = .{ 0, 0 }, .fed = 0, .logits = &.{}, .cap = 8 };
        try std.testing.expectError(error.KvFormat, restore(stream, &seq, null, &kept));
        try std.testing.expectEqual(@as(usize, 0), seq.fed);
    }
}
