//! The prompt scratch's buffers sized by use (Scratch.bytes / alloc): head_local, native_part, hc_part and part.
const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");

const hc_sk = wt.hc_sk;
const out_sk = wt.out_sk;
const att_rows = @import("cuda_batch_state.zig").att_rows;

/// head_local's rows on one GPU: the draft head's lanes (<= 16) times its ids; a split head needs a pass's rows.
pub const head_rows_one: usize = 32;

/// head_local's rows, and whether the lane reduces K slices through native_part (with clusters it never does).
pub const Sizes = struct { head_rows: usize = head_rows_one, lane_reduce: bool = true };

/// hc_part's rows of (low + 4) floats: hc_sk slices below 64 rows (legacy kernel), <= 8 below 256 (lane), one beyond.
pub fn hcPartRows(cap: usize) usize {
    return @max(cap, @max(hc_sk * @min(cap, 63), @as(usize, 8) * @min(cap, 255)));
}

/// part's rows of dims: an out projection's slice plus the zero slice, or a legacy lane tile's out_sk below 64 rows.
pub fn partRows(cap: usize) usize {
    return @max(2 * cap, out_sk * @min(cap, 63));
}

/// The rows a head piece takes at most: its fc lanes run `streams` rows a token and the decode-shaped matmul
/// zeroes their second slice into part (cuda_tiles.matmul), whose two slices hold partRows(cap) rows — past
/// this bound the fill starts one row past the allocation and the driver refuses it (CUDA_ERROR_INVALID_VALUE).
pub fn headPieceRows(cap: usize) usize {
    return @max(@as(usize, 1), partRows(cap) / (2 * wt.streams));
}

/// The block scores' row stride for a cache of `cap` tokens: its blocks rounded up to 16 (the captured _scores').
pub fn scoreStride(cap: usize) usize {
    return ((cap + wt.index_ratio - 1) / wt.index_ratio + 15) / 16 * 16;
}

test "a head piece's fc lanes fit part's two slices at every scratch size" {
    for ([_]usize{ 32, 62, 122, 252, 512, 1536, 3072, 8192, 16384 }) |cap| {
        // the fc_hidden lane runs streams rows a token and the decode-shaped matmul zeroes its second slice:
        // two slices of streams * rows a piece must fit part's partRows(cap) rows
        try std.testing.expect(2 * wt.streams * headPieceRows(cap) <= partRows(cap));
    }
    // the shipped TF_FLASHNEXT_HEAD_PIECE=1536: a 1,536-row piece (halved by the engine's chunk) must still
    // take 768 rows on the default chunk's scratch, or the head's prompt pass pays its rows in halves
    try std.testing.expectEqual(@as(usize, 768), headPieceRows(3072));
    try std.testing.expectEqual(@as(usize, 2048), headPieceRows(8192));
}

/// Bytes of block scores a pass over a cache of `cap` tokens needs for chunks of up to `rows` rows.
pub fn scoreBytes(cap: usize, rows: usize) usize {
    return @min(rows, att_rows) * scoreStride(cap) * 4;
}
