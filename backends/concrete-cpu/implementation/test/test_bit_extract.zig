const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const allocator = std.heap.page_allocator;

const common = @import("common.zig");

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

fn test3(sec_csprng: *cpu.SecCsprng, enc_csprng: *cpu.EncCsprng) !void {
    const polynomial_size: usize = 1024;
    const glwe_dim: usize = 1;
    const small_dim: usize = 585;
    const level_bsk: usize = 2;
    const base_log_bsk: usize = 10;
    const level_ksk: usize = 7;
    const base_log_ksk: usize = 4;

    const variance = std.math.pow(f64, 2, -2 * 60);

    const number_of_bits_of_message = 5;
    var raw_fft = c.aligned_alloc(cpu.CONCRETE_FFT_ALIGN, cpu.CONCRETE_FFT_SIZE);
    const fft: *cpu.Fft = @ptrCast(raw_fft);

    cpu.concrete_cpu_construct_concrete_fft(fft, polynomial_size);

    const big_dim = glwe_dim * polynomial_size;

    const small_sk = try allocator.alloc(u64, small_dim);
    cpu.concrete_cpu_init_secret_key_u64(small_sk.ptr, small_dim, sec_csprng);

    const big_sk = try allocator.alloc(u64, big_dim);
    cpu.concrete_cpu_init_secret_key_u64(big_sk.ptr, big_dim, sec_csprng);

    const bsk_f = try common.new_bsk(
        enc_csprng,
        small_dim,
        glwe_dim,
        polynomial_size,
        level_bsk,
        base_log_bsk,
        variance,
        small_sk,
        big_sk,
        fft,
    );
    defer allocator.free(bsk_f);

    const ksk_size = cpu.concrete_cpu_keyswitch_key_size_u64(level_ksk, big_dim, small_dim);

    const ksk = try allocator.alloc(u64, ksk_size);
    defer allocator.free(ksk);

    cpu.concrete_cpu_init_lwe_keyswitch_key_u64(
        ksk.ptr,
        big_sk.ptr,
        small_sk.ptr,
        big_dim,
        small_dim,
        level_ksk,
        base_log_ksk,
        variance,
        enc_csprng,
    );

    const delta_log = 64 - number_of_bits_of_message;

    // 19 in binary is 10011, so has the high bit, low bit set and is not symetrical
    const val: u64 = 19;

    std.debug.assert(1 << number_of_bits_of_message > val);
    const message = val << delta_log;

    // We will extract all bits
    const number_of_bits_to_extract = number_of_bits_of_message;

    const in_ct = try allocator.alloc(u64, big_dim + 1);
    defer allocator.free(in_ct);

    cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(
        big_sk.ptr,
        in_ct.ptr,
        message,
        big_dim,
        variance,
        enc_csprng,
    );

    const out_cts = try allocator.alloc(u64, (small_dim + 1) * number_of_bits_to_extract);
    defer allocator.free(out_cts);

    var stack_align: usize = 0;
    var stack_size: usize = 0;

    try std.testing.expect(cpu.concrete_cpu_extract_bit_lwe_ciphertext_u64_scratch(
        &stack_size,
        &stack_align,
        small_dim,
        big_dim,
        glwe_dim,
        polynomial_size,
        fft,
    ) == 0);

    const stack = @as([*]u8, @ptrCast(c.aligned_alloc(stack_align, stack_size) orelse unreachable))[0..stack_size];
    defer c.free(stack.ptr);

    cpu.concrete_cpu_extract_bit_lwe_ciphertext_u64(
        out_cts.ptr,
        in_ct.ptr,
        bsk_f.ptr,
        ksk.ptr,
        small_dim,
        number_of_bits_to_extract,
        big_dim,
        number_of_bits_to_extract,
        delta_log,
        level_bsk,
        base_log_bsk,
        glwe_dim,
        polynomial_size,
        small_dim,
        level_ksk,
        base_log_ksk,
        big_dim,
        small_dim,
        fft,
        stack.ptr,
        stack.len,
    );

    var i: u64 = 0;
    while (i < number_of_bits_to_extract) {
        const expected = (val >> @intCast(number_of_bits_of_message - 1 - i)) & 1;
        var decrypted: u64 = 0;
        cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(small_sk.ptr, out_cts[(small_dim + 1) * i ..].ptr, small_dim, &decrypted);

        const rounded = common.closest_representable(decrypted, 1, 1);
        const decoded = rounded >> 63;

        std.debug.assert(decoded == expected);
        i += 1;
    }
}

test "bit_extract" {
    var raw_sec_csprng = c.aligned_alloc(cpu.SECRET_CSPRNG_ALIGN, cpu.SECRET_CSPRNG_SIZE);
    defer c.free(raw_sec_csprng);
    const sec_csprng: *cpu.SecCsprng = @ptrCast(raw_sec_csprng);
    cpu.concrete_cpu_construct_secret_csprng(
        sec_csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_secret_csprng(sec_csprng);

    var raw_enc_csprng = c.aligned_alloc(cpu.ENCRYPTION_CSPRNG_ALIGN, cpu.ENCRYPTION_CSPRNG_SIZE);
    defer c.free(raw_enc_csprng);
    const enc_csprng: *cpu.EncCsprng = @ptrCast(raw_enc_csprng);
    cpu.concrete_cpu_construct_encryption_csprng(
        enc_csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_encryption_csprng(enc_csprng);

    try test3(sec_csprng, enc_csprng);
}
