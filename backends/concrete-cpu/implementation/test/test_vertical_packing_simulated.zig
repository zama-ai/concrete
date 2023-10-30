const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const allocator = std.heap.page_allocator;

const common = @import("common.zig");

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

fn test3() !void {
    const stdout = std.io.getStdOut().writer();
    // -ln2:   k,  N,    n, br_l,br_b, ks_l,ks_b, cb_l,cb_b, pp_l,pp_b,  cost, p_error
    // - 5 :   2, 10,  762,    5,  8,     8,  2,     3,  7,     3, 13,  10058, 6.3e-5
    const log_poly_size = 10;
    // const polynomial_size = 1 << log_poly_size;
    const glwe_dim: usize = 2;
    const small_dim: usize = 750;
    const level_cbs: usize = 6;
    const base_log_cbs: usize = 7;
    const ciphertext_modulus_log: u32 = 64;
    const security_level: u64 = 128;

    // We are going to encrypt two ciphertexts with 5 bits each
    const number_of_input_bits: usize = 14;

    // Test on 610, binary representation
    const val: u64 = 610; // 10011 00010
    // 98 ; // 00011 00010

    const one: u64 = 1;

    const extract_bits_output_buffer = try allocator.alloc(u64, number_of_input_bits);
    defer allocator.free(extract_bits_output_buffer);

    var i: u64 = 0;
    // Decryption of extracted bits for sanity check
    while (i < number_of_input_bits) {
        const bit: u64 =
            (val >> @intCast(number_of_input_bits - i - 1)) % 2;

        // cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(small_sk.ptr, extract_bits_output_buffer[(small_dim + 1) * i ..].ptr, bit << 63, small_dim, variance, csprng, &cpu.CONCRETE_CSPRNG_VTABLE);
        extract_bits_output_buffer[i] = bit << 63;
        i += 1;
    }
    // try stdout.print("bits: {any}", .{extract_bits_output_buffer});
    // We'll apply a single table look-up computing x + 1 to our 10 bits input integer that was
    // represented over two 5 bits ciphertexts
    const number_of_luts_and_output_cts: usize = 1;

    var cbs_vp_output_buffer = try allocator.alloc(u64, number_of_luts_and_output_cts);
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

    // i = 0;
    // try stdout.print("\n\n",     .{});
    // while (i < luts_length/4) {
    //     try stdout.print("{}, ", .{luts[i]});
    //     i += 1;
    // }

    {

        // const stack = @ptrCast([*]u8, c.aligned_alloc(stack_align, stack_size) orelse unreachable)[0..stack_size];
        // defer c.free(stack.ptr);

        cpu.simulation_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
            extract_bits_output_buffer.ptr,
            cbs_vp_output_buffer.ptr,
            number_of_input_bits,
            number_of_luts_and_output_cts,
            // luts_length,
            1 << number_of_input_bits,
            number_of_luts_and_output_cts,
            luts.ptr, //PolynomialList<&[u64]>,
            glwe_dim,
            log_poly_size,
            small_dim,
            level_cbs,
            base_log_cbs,
            ciphertext_modulus_log,
            security_level,
        );
    }

    const expected = val + 1;

    var decrypted: u64 = cbs_vp_output_buffer[0];
    // cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(big_sk.ptr, cbs_vp_output_buffer.ptr, big_dim, &decrypted);
    try stdout.print("decrypted {}\n", .{decrypted});
    const rounded =
        common.closest_representable(decrypted, 1, number_of_input_bits);
    const decoded = rounded >> delta_log_lut;

    try stdout.print("decoded {}\n", .{decoded});
    try stdout.print("expected {}\n", .{expected});

    std.debug.assert(decoded == expected);
}

test "vertical_packing_simulated" {
    // var raw_csprng = c.aligned_alloc(cpu.CONCRETE_CSPRNG_ALIGN, cpu.CONCRETE_CSPRNG_SIZE);
    // defer c.free(raw_csprng);
    // const csprng = @ptrCast(*cpu.Csprng, raw_csprng);
    // cpu.concrete_cpu_construct_concrete_csprng(
    //     csprng,
    //     cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    // );
    // defer cpu.concrete_cpu_destroy_concrete_csprng(csprng);

    // //CMUX tree
    // try test3(csprng, 512);
    //
    // //No CMUX tree
    // try test3(csprng, 1024);
    //
    // //Expanded lut
    // try test3(csprng, 2048);
    // std.debug.
    try test3();
}
