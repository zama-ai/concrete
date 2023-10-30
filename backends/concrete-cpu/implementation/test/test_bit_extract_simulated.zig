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
    const log_poly_size: usize = 10;
    // const polynomial_size: u64 = 1 << log_poly_size;
    const glwe_dim: u64 = 1;
    const small_dim: u64 = 585;
    const level_bsk: u64 = 2;
    const base_log_bsk: u64 = 10;
    const level_ksk: u64 = 7;
    const base_log_ksk: u64 = 4;
    const ciphertext_modulus_log: u32 = 64;
    const security_level: u64 = 128;

    // const variance = std.math.pow(f64, 2, -2 * 60);

    const number_of_bits_of_message = 5;

    // const big_dim = glwe_dim * polynomial_size;

    const delta_log = 64 - number_of_bits_of_message;

    // 19 in binary is 10011, so has the high bit, low bit set and is not symetrical
    const val: u64 = 19;

    std.debug.assert(1 << number_of_bits_of_message > val);
    const message = val << delta_log;

    // We will extract all bits
    const number_of_bits_to_extract = number_of_bits_of_message;

    const out_cts = try allocator.alloc(u64, number_of_bits_to_extract);
    defer allocator.free(out_cts);

    cpu.simulation_extract_bit_lwe_ciphertext_u64(
        out_cts.ptr,
        message,
        delta_log,
        number_of_bits_to_extract,
        log_poly_size,
        glwe_dim,
        small_dim,
        base_log_ksk,
        level_ksk,
        base_log_bsk,
        level_bsk,
        ciphertext_modulus_log,
        security_level,
    );

    var i: u64 = 0;
    while (i < number_of_bits_to_extract) {
        const expected = (val >> @intCast(number_of_bits_of_message - 1 - i)) & 1;
        var decrypted: u64 = out_cts[i];
        // cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(small_sk.ptr, out_cts[(small_dim + 1) * i ..].ptr, lwe_dimension, &decrypted);

        const rounded = common.closest_representable(decrypted, 1, 1);
        const decoded = rounded >> 63;
        const stdout = std.io.getStdOut().writer();
        try stdout.print("decoded {}\n", .{decoded});
        try stdout.print("expected {}\n", .{expected});
        // std.debug.print(decoded, .{});
        // std.debug.print(expected, .{});
        std.debug.assert(decoded == expected);
        i += 1;
    }
}

test "bit_extract_simu" {
    try test3();
}
