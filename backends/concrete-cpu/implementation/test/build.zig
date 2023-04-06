const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const mode = b.standardReleaseOptions();

    const test_step = b.step("test", "Run library tests");

    const test_ = b.addTest("all.zig");

    test_.setBuildMode(mode);
    test_.linkLibC();
    test_.linkSystemLibraryName("unwind");
    test_.linkSystemLibraryName("concrete_cpu");
    test_.addLibraryPath("../target/debug");

    test_.addIncludePath("../include");

    test_step.dependOn(&test_.step);
}
