use criterion::{black_box, criterion_group, criterion_main, Criterion};
use v0_parameters::{all_results, Args, MAX_LWE_DIM, MIN_LWE_DIM, _4_SIGMA};

fn v0_pbs_optimization(c: &mut Criterion) {
    let args: Args = Args {
        min_precision: 1,
        max_precision: 8,
        p_error: _4_SIGMA,
        security_level: 128,
        min_log_poly_size: 8,
        max_log_poly_size: 16,
        min_glwe_dim: 1,
        max_glwe_dim: 1,
        min_intern_lwe_dim: MIN_LWE_DIM,
        max_intern_lwe_dim: MAX_LWE_DIM,
        sum_size: 4096,
        no_parallelize: true,
        wop_pbs: false,
        simulate_dag: false,
        cache_on_disk: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
    };

    c.bench_function("v0 PBS table generation", |b| {
        b.iter(|| black_box(all_results(&args)))
    });
}

fn v0_pbs_optimization_simulate_graph(c: &mut Criterion) {
    let args: Args = Args {
        min_precision: 1,
        max_precision: 8,
        p_error: _4_SIGMA,
        security_level: 128,
        min_log_poly_size: 8,
        max_log_poly_size: 16,
        min_glwe_dim: 1,
        max_glwe_dim: 1,
        min_intern_lwe_dim: MIN_LWE_DIM,
        max_intern_lwe_dim: MAX_LWE_DIM,
        sum_size: 4096,
        no_parallelize: true,
        wop_pbs: false,
        simulate_dag: true,
        cache_on_disk: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
    };

    c.bench_function("v0 PBS simulate dag table generation", |b| {
        b.iter(|| black_box(all_results(&args)))
    });
}

fn v0_wop_pbs_optimization(c: &mut Criterion) {
    let args = Args {
        min_precision: 1,
        max_precision: 16,
        p_error: _4_SIGMA,
        security_level: 128,
        min_log_poly_size: 10,
        max_log_poly_size: 11,
        min_glwe_dim: 1,
        max_glwe_dim: 2,
        min_intern_lwe_dim: 450,
        max_intern_lwe_dim: 600,
        sum_size: 4096,
        no_parallelize: true,
        wop_pbs: true,
        simulate_dag: false,
        cache_on_disk: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
    };

    c.bench_function("v0 WoP-PBS table generation", |b| {
        b.iter(|| black_box(all_results(&args)))
    });
}

criterion_group!(
    benches,
    v0_pbs_optimization,
    v0_pbs_optimization_simulate_graph,
    v0_wop_pbs_optimization
);
criterion_main!(benches);
