const c = @cImport({
    @cInclude("stdlib.h");
});

const std = @import("std");

const allocator = std.heap.page_allocator;

const random = std.rand.Random;

const common = @import("common.zig");

const cpu = @cImport({
    @cInclude("concrete-cpu.h");
});

const KeySet = struct {
    in_dim: u64,
    glwe_dim: u64,
    polynomial_size: u64,
    level: u64,
    base_log: u64,
    in_sk: []u64,
    out_sk: []u64,
    bsk_f: []f64,
    fft: *cpu.Fft,
    stack: []u8,

    pub fn init(
        csprng: *cpu.Csprng,
        in_dim: u64,
        glwe_dim: u64,
        polynomial_size: u64,
        level: u64,
        base_log: u64,
        key_variance: f64,
    ) !KeySet {
        var raw_fft = c.aligned_alloc(cpu.CONCRETE_FFT_ALIGN, cpu.CONCRETE_FFT_SIZE);
        const fft = @ptrCast(*cpu.Fft, raw_fft);

        cpu.concrete_cpu_construct_concrete_fft(fft, polynomial_size);

        const out_dim = glwe_dim * polynomial_size;

        const in_sk = try allocator.alloc(u64, in_dim);
        cpu.concrete_cpu_init_secret_key_u64(in_sk.ptr, in_dim, csprng, &cpu.CONCRETE_CSPRNG_VTABLE);

        const out_sk = try allocator.alloc(u64, out_dim);
        cpu.concrete_cpu_init_secret_key_u64(out_sk.ptr, out_dim, csprng, &cpu.CONCRETE_CSPRNG_VTABLE);

        const bsk_f = try common.new_bsk(
            csprng,
            in_dim,
            glwe_dim,
            polynomial_size,
            level,
            base_log,
            key_variance,
            in_sk,
            out_sk,
            fft,
        );

        var stack_size: usize = 0;
        var stack_align: usize = 0;
        try std.testing.expect(
            cpu.concrete_cpu_bootstrap_lwe_ciphertext_u64_scratch(
                &stack_size,
                &stack_align,
                glwe_dim,
                polynomial_size,
                fft,
            ) == 0,
        );
        const stack = @ptrCast([*]u8, c.aligned_alloc(stack_align, stack_size))[0..stack_size];

        return KeySet{
            .in_dim = in_dim,
            .glwe_dim = glwe_dim,
            .polynomial_size = polynomial_size,
            .level = level,
            .base_log = base_log,
            .in_sk = in_sk,
            .out_sk = out_sk,
            .bsk_f = bsk_f,
            .fft = fft,
            .stack = stack,
        };
    }

    pub fn bootstrap(
        self: *KeySet,
        pt: u64,
        encryption_variance: f64,
        lut: []u64,
        csprng: *cpu.Csprng,
    ) !u64 {
        const out_dim = self.glwe_dim * self.polynomial_size;

        const in_ct = try allocator.alloc(u64, self.in_dim + 1);
        defer allocator.free(in_ct);

        const out_ct = try allocator.alloc(u64, out_dim + 1);
        defer allocator.free(out_ct);

        cpu.concrete_cpu_encrypt_lwe_ciphertext_u64(
            self.in_sk.ptr,
            in_ct.ptr,
            pt,
            self.in_dim,
            encryption_variance,
            csprng,
            &cpu.CONCRETE_CSPRNG_VTABLE,
        );

        cpu.concrete_cpu_bootstrap_lwe_ciphertext_u64(
            out_ct.ptr,
            in_ct.ptr,
            lut.ptr,
            self.bsk_f.ptr,
            self.level,
            self.base_log,
            self.glwe_dim,
            self.polynomial_size,
            self.in_dim,
            self.fft,
            self.stack.ptr,
            self.stack.len,
        );

        var image: u64 = 0;
        cpu.concrete_cpu_decrypt_lwe_ciphertext_u64(self.out_sk.ptr, out_ct.ptr, out_dim, &image);
        return image;
    }

    pub fn deinit(
        self: *KeySet,
    ) void {
        allocator.free(self.in_sk);
        allocator.free(self.out_sk);
        allocator.free(self.bsk_f);
        cpu.concrete_cpu_destroy_concrete_fft(self.fft);
        c.free(self.fft);
        c.free(self.stack.ptr);
    }
};

fn expand_lut(lut: []u64, glwe_dim: u64, polynomial_size: u64) ![]u64 {
    const raw_lut = try allocator.alloc(u64, (glwe_dim + 1) * polynomial_size);

    std.debug.assert(polynomial_size % lut.len == 0);
    const lut_case_size = polynomial_size / lut.len;

    for (raw_lut[0..(glwe_dim * polynomial_size)]) |*i| {
        i.* = 0;
    }

    var i: usize = 0;

    while (i < lut.len) {
        var j: usize = 0;
        while (j < lut_case_size) {
            raw_lut[glwe_dim * polynomial_size + i * lut_case_size + j] = lut[i];
            j += 1;
        }
        i += 1;
    }
    return raw_lut;
}

fn encrypt_bootstrap_decrypt(
    csprng: *cpu.Csprng,
    lut: []u64,
    lut_index: u64,
    in_dim: u64,
    glwe_dim: u64,
    polynomial_size: u64,
    level: u64,
    base_log: u64,
    key_variance: f64,
    encryption_variance: f64,
) !u64 {
    const precision = lut.len;

    var key_set = try KeySet.init(
        csprng,
        in_dim,
        glwe_dim,
        polynomial_size,
        level,
        base_log,
        key_variance,
    );
    defer key_set.deinit();

    const raw_lut = try expand_lut(lut, glwe_dim, polynomial_size);
    defer allocator.free(raw_lut);

    const pt = (@intToFloat(f64, lut_index) + 0.5) / (2.0 * @intToFloat(f64, precision)) * std.math.pow(f64, 2.0, 64);

    const image = try key_set.bootstrap(@floatToInt(u64, pt), encryption_variance, raw_lut, csprng);

    return image;
}

test "bootstrap" {
    var raw_csprng = c.aligned_alloc(cpu.CONCRETE_CSPRNG_ALIGN, cpu.CONCRETE_CSPRNG_SIZE);
    defer c.free(raw_csprng);
    const csprng = @ptrCast(*cpu.Csprng, raw_csprng);
    cpu.concrete_cpu_construct_concrete_csprng(
        csprng,
        cpu.Uint128{ .little_endian_bytes = [_]u8{1} ** 16 },
    );
    defer cpu.concrete_cpu_destroy_concrete_csprng(csprng);

    const log2_precision = 4;
    const precision = 1 << log2_precision;

    const lut = try allocator.alloc(u64, precision);
    defer allocator.free(lut);
    try std.os.getrandom(std.mem.sliceAsBytes(lut));

    var lut_index: u64 = 0;
    try std.os.getrandom(std.mem.asBytes(&lut_index));
    lut_index %= 2 * precision;

    const in_dim = 3;
    const glwe_dim = 1;
    const log2_poly_size = 10;
    const polynomial_size = 1 << log2_poly_size;
    const level = 3;
    const base_log = 10;
    const key_variance = 0.0000000000000000000001;
    const encryption_variance = 0.0000000000000000000001;

    const image = try encrypt_bootstrap_decrypt(csprng, lut, lut_index, in_dim, glwe_dim, polynomial_size, level, base_log, key_variance, encryption_variance);

    const expected_image = if (lut_index < precision) lut[lut_index] else -%lut[(lut_index - precision)];

    const diff = @intToFloat(f64, @bitCast(i64, image -% expected_image)) / std.math.pow(f64, 2.0, 64);

    try std.testing.expect(@fabs(diff) < 0.001);
}
