const std = @import("std");
// -pub fn build(b: *std.build.Builder) void {
// -    const mode = b.standardReleaseOptions();
// -    const target = b.standardTargetOptions(.{});
// +pub fn build(b: *std.Build) void {
// +    const optimize = b.standardOptimizeOption(.{});
// +    const target = b.standardTargetOptions(.{});
// pub fn build(b: *std.build.Builder) void {
pub fn build(b: *std.Build) void {
    // const mode = b.standardReleaseOptions();
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});
    const test_step = b.step("test", "Run library tests");

    const test_ = b.addTest(.{
        .name = "all",
        .root_source_file=.{.path = "all.zig"},
        .target = target,
        .optimize = optimize,
    }
    );

    // "all.zig"
    // -const exe = b.addExecutable("example", "src/main.zig");
    // -exe.setBuildMode(mode);
    // -exe.setTarget(target);
    // +const exe = b.addExecutable(.{
    // +    .name = "example",
    // +    .root_source_file = "src/main.zig",
    // +    .target = target,
    // +    .optimize = optimize,
    // +});

    // test_.setBuildMode(optimize);
    // test_.setBuildMode(mode);
    test_.linkLibC();
    test_.linkSystemLibraryName("unwind");
    test_.linkSystemLibraryName("concrete_cpu");
    test_.addLibraryPath("../target/debug");

    test_.addIncludePath("../include");

    test_.linkFramework("Security");

    test_step.dependOn(&test_.step);
}
