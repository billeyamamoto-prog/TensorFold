//! The GLP control vector reader's host checks need no GPU, no model download and no checkpoint.
const std = @import("std");

pub fn tests(b: *std.Build, all: *std.Build.Step) void {
    const target = b.graph.host;
    const step = b.step("test-glp", "Control vector container, metadata rules and refusals");
    all.dependOn(step);
    const module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/glp.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    });
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    host.setCwd(b.path("."));
    step.dependOn(&host.step);

    // The device-free shape rules of the CUDA lane's plan: whole-stream width, one steer a layer, unit rows.
    const steer = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/flashnext/steer_source.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    }) }));
    step.dependOn(&steer.step);

    // The serve startup's validation: the --glp refusals against the same reader (port: wired so
    // the checks run on the host; the GLP branch carried the test file without a build wiring).
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const api = b.createModule(.{
        .root_source_file = b.path("zig/src/core/engine_api.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const startup = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/glp_startup_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{.{ .name = "engine_api", .module = api }},
    }) }));
    step.dependOn(&startup.step);

    // The contract's steer reference: the float64 arithmetic rules the kernel must meet, and the
    // steering launch's limits (port: wired so the checks run on the host; the wrapper sits at
    // core so the contract files' parent-relative imports resolve).
    const steer_ref = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/steer_contract_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    }) }));
    step.dependOn(&steer_ref.step);

    // The checker: the reader's view of a file, and a rewrite of it, for another implementation to compare against.
    const tool = b.addExecutable(.{ .name = "tf-glp-check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/glp_check.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "glp", .module = module }} }) });
    b.step("tf-glp-check", "Show a control vector file as this reader sees it").dependOn(&b.addInstallArtifact(tool, .{}).step);
}
