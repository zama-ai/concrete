use std::time::Duration;

use concrete_cpu::c_api::linear_op::{
    concrete_cpu_add_lwe_ciphertext_u64, concrete_cpu_add_plaintext_lwe_ciphertext_u64,
    concrete_cpu_mul_cleartext_lwe_ciphertext_u64, concrete_cpu_negate_lwe_ciphertext_u64,
};
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};

pub fn linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(100));

    for lwe_dimension in [128, 256, 512] {
        let lwe_size = lwe_dimension + 1;
        group.bench_function(&format!("add-lwe-ciphertext-u64-{lwe_dimension}"), |b| {
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

        group.bench_function(&format!("add-lwe-plaintext-u64-{lwe_dimension}"), |b| {
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

        group.bench_function(&format!("mul-lwe-cleartext-u64-{lwe_dimension}"), |b| {
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

        group.bench_function(&format!("negate-lwe-ciphertext-u64-{lwe_dimension}"), |b| {
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

criterion_group!(benches, linear);
criterion_main!(benches);
