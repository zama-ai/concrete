const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const cpu = @cImport({
    @cInclude("include/concrete-cpu.h");
});

const allocator = std.heap.page_allocator;

fn test_encrypt_decrypt(csprng: *cpu.Csprng, pt: u64, dim: u64) !u64 {
    const sk = try allocator.alloc(u64, dim);
    defer allocator.free(sk);

    cpu.concrete_cpu_init_lwe_secret_key_u64(sk.ptr, dim, csprng, &cpu.CONCRETE_CSPRNG_VTABLE);

    const ct = try allocator.alloc(u64, dim + 1);
    defer allocator.free(ct);

    cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(sk.ptr, ct.ptr, pt, dim, 0.000000000000001, csprng, &cpu.CONCRETE_CSPRNG_VTABLE);

    var result: u64 = 0;

    cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(sk.ptr, ct.ptr, dim, &result);

    return result;
}

test "encryption" {
    var raw_csprng = c.aligned_alloc(cpu.CONCRETE_CSPRNG_ALIGN, cpu.CONCRETE_CSPRNG_SIZE);
    defer c.free(raw_csprng);
    const csprng = @ptrCast(*cpu.Csprng, raw_csprng);
    cpu.concrete_cpu_construct_concrete_csprng(
        csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_concrete_csprng(csprng);

    const pt = 1 << 63;

    const result = try test_encrypt_decrypt(csprng, pt, 1024);

    const diff = @intToFloat(f64, @bitCast(i64, result -% pt)) / std.math.pow(f64, 2.0, 64);

    try std.testing.expect(@fabs(diff) < 0.001);
}
