use criterion::{black_box, criterion_group, criterion_main, Criterion};
use v0_parameters::{all_results, Args, MAX_LWE_DIM, MIN_LWE_DIM, _4_SIGMA};

fn pbs_benchmark(c: &mut Criterion) {
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
    };

    c.bench_function("PBS table generation", |b| {
        b.iter(|| black_box(all_results(&args)))
    });
}

fn wop_pbs_benchmark(c: &mut Criterion) {
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
    };

    c.bench_function("WoP-PBS table generation", |b| {
        b.iter(|| black_box(all_results(&args)))
    });
}

criterion_group!(benches, pbs_benchmark, wop_pbs_benchmark);
criterion_main!(benches);
