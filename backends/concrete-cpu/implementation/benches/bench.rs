use concrete_cpu::c_api::linear_op::{
    concrete_cpu_add_lwe_ciphertext_u64, concrete_cpu_add_plaintext_lwe_ciphertext_u64,
    concrete_cpu_mul_cleartext_lwe_ciphertext_u64, concrete_cpu_negate_lwe_ciphertext_u64,
};
use concrete_cpu::implementation::polynomial::update_with_wrapping_add_mul;
#[cfg(target_arch = "x86_64")]
use concrete_cpu::implementation::polynomial::update_with_wrapping_add_mul_ntt;
use concrete_cpu::implementation::types::polynomial::Polynomial;

#[cfg(target_arch = "x86_64")]
use concrete_ntt::native_binary64::Plan32;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    for lwe_dimension in [128, 256, 512] {
        let lwe_size = lwe_dimension + 1;
        c.bench_function(&format!("add-lwe-ciphertext-u64-{lwe_dimension}"), |b| {
            let mut out = vec![0_u64; lwe_size];
            let ct0 = vec![0_u64; lwe_size];
            let ct1 = vec![0_u64; lwe_size];
            b.iter(|| unsafe {
                concrete_cpu_add_lwe_ciphertext_u64(
                    out.as_mut_ptr(),
                    ct0.as_ptr(),
                    ct1.as_ptr(),
                    lwe_dimension,
                );
            });
        });

        c.bench_function(&format!("add-lwe-plaintext-u64-{lwe_dimension}"), |b| {
            let mut out = vec![0_u64; lwe_size];
            let ct0 = vec![0_u64; lwe_size];
            let plaintext = 0_u64;
            b.iter(|| unsafe {
                concrete_cpu_add_plaintext_lwe_ciphertext_u64(
                    out.as_mut_ptr(),
                    ct0.as_ptr(),
                    plaintext,
                    lwe_dimension,
                );
            });
        });

        c.bench_function(&format!("mul-lwe-cleartext-u64-{lwe_dimension}"), |b| {
            let mut out = vec![0_u64; lwe_size];
            let ct0 = vec![0_u64; lwe_size];
            let cleartext = 0_u64;
            b.iter(|| unsafe {
                concrete_cpu_mul_cleartext_lwe_ciphertext_u64(
                    out.as_mut_ptr(),
                    ct0.as_ptr(),
                    cleartext,
                    lwe_dimension,
                );
            });
        });

        c.bench_function(&format!("negate-lwe-ciphertext-u64-{lwe_dimension}"), |b| {
            let mut out = vec![0_u64; lwe_size];
            let ct0 = vec![0_u64; lwe_size];
            b.iter(|| unsafe {
                concrete_cpu_negate_lwe_ciphertext_u64(
                    out.as_mut_ptr(),
                    ct0.as_ptr(),
                    lwe_dimension,
                );
            });
        });
    }
}

pub fn polynomial_multiplication(c: &mut Criterion) {
    for log_poly_size in 8..15 {
        let poly_size = 1 << log_poly_size;

        let input: Vec<u64> = (0..poly_size as u64).collect();
        let input = Polynomial::new(input.as_slice(), poly_size);

        let input_bin: Vec<u64> = (0..poly_size / 2)
            .flat_map(|_| [0, 1].into_iter())
            .collect();
        let input2 = Polynomial::new(input_bin.as_slice(), poly_size);

        let mut output: Vec<u64> = (0..poly_size as u64).collect();
        let mut output = Polynomial::new(output.as_mut_slice(), poly_size);

        #[cfg(target_arch = "x86_64")]
        let mut buffer: Vec<u64> = (0..poly_size as u64).collect();

        #[cfg(target_arch = "x86_64")]
        let mut buffer = Polynomial::new(buffer.as_mut_slice(), poly_size);

        #[cfg(target_arch = "x86_64")]
        let plan = Plan32::try_new(poly_size).unwrap();

        c.bench_function(&format!("poly-mult-{}", poly_size), |b| {
            b.iter(|| update_with_wrapping_add_mul(output.as_mut_view(), input, input2));
        });
        c.bench_function(&format!("ntt-poly-mult-{}", poly_size), |b| {
            b.iter(|| {
                #[cfg(target_arch = "x86_64")]
                update_with_wrapping_add_mul_ntt(
                    &plan,
                    output.as_mut_view(),
                    input,
                    input2,
                    buffer.as_mut_view(),
                );
                #[cfg(not(target_arch = "x86_64"))]
                update_with_wrapping_add_mul(output.as_mut_view(), input, input2);
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark, polynomial_multiplication);
criterion_main!(benches);
