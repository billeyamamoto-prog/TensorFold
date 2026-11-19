//! Prompt-pass switches read once at load (environment).
const std = @import("std");

/// TF_FLASHNEXT_PROMPT_MX128=N: MXFP8 projections N columns wide or more take the relaxed 128 x 128 MXFP8 tile in prompt
/// passes (default 1024); 0 keeps the exact kernels everywhere. The tile's rows are not the exact kernels' bits.
pub fn mx128MinN() usize {
    const v = std.c.getenv("TF_FLASHNEXT_PROMPT_MX128") orelse return 1024;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1024;
}

test "unset, the MXFP8 projections of 1,024 columns or more take the prompt tile" {
    if (std.c.getenv("TF_FLASHNEXT_PROMPT_MX128") == null) try std.testing.expectEqual(@as(usize, 1024), mx128MinN());
}
