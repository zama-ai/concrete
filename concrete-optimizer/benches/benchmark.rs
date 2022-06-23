use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
use concrete_optimizer::pareto::{extract_br_pareto, extract_ks_pareto};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn br_pareto_benchmark(c: &mut Criterion) {
    c.bench_function("BR pareto", |b| {
        b.iter(|| {
            black_box(extract_br_pareto(
                128,
                &DEFAUT_DOMAINS.glwe_pbs_constrained,
                &DEFAUT_DOMAINS.free_glwe.into(),
                64,
            ))
        })
    });
}

fn ks_pareto_benchmark(c: &mut Criterion) {
    c.bench_function("KS pareto", |b| {
        b.iter(|| {
            black_box(extract_ks_pareto(
                128,
                &DEFAUT_DOMAINS.glwe_pbs_constrained,
                &DEFAUT_DOMAINS.free_glwe.into(),
                64,
            ))
        })
    });
}

criterion_group!(benches, br_pareto_benchmark, ks_pareto_benchmark);
criterion_main!(benches);
