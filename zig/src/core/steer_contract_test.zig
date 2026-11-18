//! The GLP steering reference's contract checks, on the host: this wrapper only pulls the
//! contract/steer files into a test build rooted at core, so their parent-relative imports
//! (`../bf16_round.zig`, `../norms_ref.zig`, `../act_ref.zig`) resolve.

test {
    _ = @import("contract/steer.zig");
    _ = @import("contract/steer_records.zig");
}
