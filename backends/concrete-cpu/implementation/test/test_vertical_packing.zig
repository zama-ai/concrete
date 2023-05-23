const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const allocator = std.heap.page_allocator;

const common = @import("common.zig");

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

fn test3(sec_csprng: *cpu.SecCsprng, enc_csprng: *cpu.EncCsprng, polynomial_size: usize) !void {
    const glwe_dim: usize = 1;
    const small_dim: usize = 4;
    const level_bsk: usize = 4;
    const base_log_bsk: usize = 9;
    const level_pksk: usize = 2;
    const base_log_pksk: usize = 15;
    const level_cbs: usize = 4;
    const base_log_cbs: usize = 6;

    const variance: f64 = std.math.pow(f64, 2.0, -100);

    const big_dim = glwe_dim * polynomial_size;

    const small_sk = try allocator.alloc(u64, small_dim);
    cpu.concrete_cpu_init_secret_key_u64(small_sk.ptr, small_dim, sec_csprng);

    const big_sk = try allocator.alloc(u64, big_dim);
    cpu.concrete_cpu_init_secret_key_u64(big_sk.ptr, big_dim, sec_csprng);

    var raw_fft = c.aligned_alloc(cpu.CONCRETE_FFT_ALIGN, cpu.CONCRETE_FFT_SIZE);
    const fft = @ptrCast(*cpu.Fft, raw_fft);

    cpu.concrete_cpu_construct_concrete_fft(fft, polynomial_size);

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

    const cbs_pfpksk_size = cpu.concrete_cpu_lwe_packing_keyswitch_key_size(glwe_dim, polynomial_size, level_pksk, big_dim);

    const cbs_pfpksk = try allocator.alloc(u64, cbs_pfpksk_size * (glwe_dim + 1));
    defer allocator.free(cbs_pfpksk);

    cpu.concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
        cbs_pfpksk.ptr,
        big_sk.ptr,
        big_sk.ptr,
        big_dim,
        polynomial_size,
        glwe_dim,
        level_pksk,
        base_log_pksk,
        variance,
        1,
        enc_csprng,
    );

    // We are going to encrypt two ciphertexts with 5 bits each
    const number_of_input_bits: usize = 10;

    // Test on 610, binary representation 10011 00010
    const val: u64 = 610;
    const one: u64 = 1;

    const extract_bits_output_buffer = try allocator.alloc(u64, number_of_input_bits * (small_dim + 1));
    defer allocator.free(extract_bits_output_buffer);

    var i: u64 = 0;
    // Decryption of extracted bits for sanity check
    while (i < number_of_input_bits) {
        const bit: u64 =
            (val >> @intCast(u6, number_of_input_bits - i - 1)) % 2;

        cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(small_sk.ptr, extract_bits_output_buffer[(small_dim + 1) * i ..].ptr, bit << 63, small_dim, variance, enc_csprng);

        i += 1;
    }

    // We'll apply a single table look-up computing x + 1 to our 10 bits input integer that was
    // represented over two 5 bits ciphertexts
    const number_of_luts_and_output_cts: usize = 1;

    var cbs_vp_output_buffer = try allocator.alloc(u64, (big_dim + 1) * number_of_luts_and_output_cts);
    defer allocator.free(cbs_vp_output_buffer);

    // Here we will create a single lut containing a single polynomial, which will result in a single
    // Output ciphertecct

    const luts_length = number_of_luts_and_output_cts * (1 << number_of_input_bits);

    var luts = try allocator.alloc(u64, luts_length);
    defer allocator.free(luts);

    const delta_log_lut = 64 - number_of_input_bits;

    i = 0;
    while (i < luts_length) {
        luts[i] = ((i + 1) % (one << number_of_input_bits)) << delta_log_lut;
        i += 1;
    }

    {
        var stack_align: usize = 0;
        var stack_size: usize = 0;

        try std.testing.expect(cpu.concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64_scratch(
            &stack_size,
            &stack_align,
            number_of_luts_and_output_cts,
            small_dim,
            number_of_input_bits,
            1 << number_of_input_bits,
            number_of_luts_and_output_cts,
            glwe_dim,
            polynomial_size,
            polynomial_size,
            level_cbs,
            fft,
        ) == 0);

        const stack = @ptrCast([*]u8, c.aligned_alloc(stack_align, stack_size) orelse unreachable)[0..stack_size];
        defer c.free(stack.ptr);

        cpu.concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
            cbs_vp_output_buffer.ptr,
            extract_bits_output_buffer.ptr,
            luts.ptr,
            bsk_f.ptr,
            cbs_pfpksk.ptr,
            big_dim,
            number_of_luts_and_output_cts,
            small_dim,
            number_of_input_bits,
            1 << number_of_input_bits,
            number_of_luts_and_output_cts,
            level_bsk,
            base_log_bsk,
            glwe_dim,
            polynomial_size,
            small_dim,
            level_pksk,
            base_log_pksk,
            big_dim,
            glwe_dim,
            polynomial_size,
            glwe_dim + 1,
            level_cbs,
            base_log_cbs,
            fft,
            stack.ptr,
            stack_size,
        );
    }

    const expected = val + 1;

    var decrypted: u64 = 0;
    cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(big_sk.ptr, cbs_vp_output_buffer.ptr, big_dim, &decrypted);

    const rounded =
        common.closest_representable(decrypted, 1, number_of_input_bits);
    const decoded = rounded >> delta_log_lut;
    std.debug.assert(decoded == expected);
}

test "vertical_packing" {
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

    //CMUX tree
    try test3(sec_csprng, enc_csprng, 512);

    //No CMUX tree
    try test3(sec_csprng, enc_csprng, 1024);

    //Expanded lut
    try test3(sec_csprng, enc_csprng, 2048);
}
