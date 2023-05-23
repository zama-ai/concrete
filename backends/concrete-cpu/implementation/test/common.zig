const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const allocator = std.heap.page_allocator;

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

pub fn new_bsk(
    csprng: *cpu.EncCsprng,
    in_dim: usize,
    glwe_dim: u64,
    polynomial_size: u64,
    level: u64,
    base_log: u64,
    key_variance: f64,
    in_sk: []u64,
    out_sk: []u64,
    fft: *cpu.Fft,
) ![]cpu.c_double_complex {
    const bsk_size = cpu.concrete_cpu_bootstrap_key_size_u64(level, glwe_dim, polynomial_size, in_dim);
    const bsk = try allocator.alloc(u64, bsk_size);
    defer allocator.free(bsk);

    cpu.concrete_cpu_init_lwe_bootstrap_key_u64(
        bsk.ptr,
        in_sk.ptr,
        out_sk.ptr,
        in_dim,
        polynomial_size,
        glwe_dim,
        level,
        base_log,
        key_variance,
        1,
        csprng,
    );

    const bsk_f_size = cpu.concrete_cpu_fourier_bootstrap_key_size_u64(level, glwe_dim, polynomial_size, in_dim);
    const bsk_f = try allocator.alloc(cpu.c_double_complex, bsk_f_size);
    {
        var stack_size: u64 = 0;
        var stack_align: u64 = 0;
        try std.testing.expect(
            cpu.concrete_cpu_bootstrap_key_convert_u64_to_fourier_scratch(
                &stack_size,
                &stack_align,
                fft,
            ) == 0,
        );

        const stack = @ptrCast([*]u8, c.aligned_alloc(stack_align, stack_size) orelse unreachable)[0..stack_size];
        defer c.free(stack.ptr);

        cpu.concrete_cpu_bootstrap_key_convert_u64_to_fourier(
            bsk.ptr,
            bsk_f.ptr,
            level,
            base_log,
            glwe_dim,
            polynomial_size,
            in_dim,
            fft,
            stack.ptr,
            stack.len,
        );
    }

    return bsk_f;
}

pub fn closest_representable(input: u64, level_count: u64, base_log: u64) u64 {
    // The closest number representable by the decomposition can be computed by performing
    // the rounding at the appropriate bit.

    // We compute the number of least significant bits which can not be represented by the
    // decomposition
    const non_rep_bit_count: u64 = 64 - (level_count * base_log);
    // We generate a mask which captures the non representable bits
    const one: u64 = 1;
    const non_rep_mask = one << @intCast(u6, non_rep_bit_count - 1);
    // We retrieve the non representable bits
    const non_rep_bits = input & non_rep_mask;
    // We extract the msb of the  non representable bits to perform the rounding
    const non_rep_msb = non_rep_bits >> @intCast(u6, non_rep_bit_count - 1);
    // We remove the non-representable bits and perform the rounding
    var res = input >> @intCast(u6, non_rep_bit_count);
    res += non_rep_msb;
    return res << @intCast(u6, non_rep_bit_count);
}

pub fn highest_bits(encoded: u64) ![]u8 {
    const precision = 11;

    var buffer = try allocator.alloc(u8, precision + 2);

    const one: u64 = 1;

    const high_bits = (encoded +% (one << @intCast(u6, 64 - precision))) >> @intCast(u6, 64 - precision);

    return std.fmt.bufPrint(buffer, "0.{b:0>11}", .{high_bits});
}
