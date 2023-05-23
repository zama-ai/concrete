const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

const allocator = std.heap.page_allocator;

fn test_encrypt_decrypt(sec_csprng: *cpu.SecCsprng, enc_csprng: *cpu.EncCsprng, pt: u64, dim: u64) !u64 {
    const sk = try allocator.alloc(u64, dim);
    defer allocator.free(sk);

    cpu.concrete_cpu_init_secret_key_u64(sk.ptr, dim, sec_csprng);

    const ct = try allocator.alloc(u64, dim + 1);
    defer allocator.free(ct);

    cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(sk.ptr, ct.ptr, pt, dim, 0.000000000000001, enc_csprng);

    var result: u64 = 0;

    cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(sk.ptr, ct.ptr, dim, &result);

    return result;
}

test "encryption" {
    var raw_sec_csprng = c.aligned_alloc(cpu.SECRET_CSPRNG_ALIGN, cpu.SECRET_CSPRNG_SIZE);
    defer c.free(raw_sec_csprng);
    const sec_csprng = @ptrCast(*cpu.SecCsprng, raw_sec_csprng);
    cpu.concrete_cpu_construct_secret_csprng(
        sec_csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_secret_csprng(sec_csprng);

    var raw_enc_csprng = c.aligned_alloc(cpu.ENCRYPTION_CSPRNG_ALIGN, cpu.ENCRYPTION_CSPRNG_SIZE);
    defer c.free(raw_enc_csprng);
    const enc_csprng = @ptrCast(*cpu.EncCsprng, raw_enc_csprng);
    cpu.concrete_cpu_construct_encryption_csprng(
        enc_csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_encryption_csprng(enc_csprng);

    const pt = 1 << 63;

    const result = try test_encrypt_decrypt(sec_csprng, enc_csprng, pt, 1024);

    const diff = @intToFloat(f64, @bitCast(i64, result -% pt)) / std.math.pow(f64, 2.0, 64);

    try std.testing.expect(@fabs(diff) < 0.001);
}
