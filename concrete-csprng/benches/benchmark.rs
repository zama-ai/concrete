use concrete_csprng::RandomGenerator;
use criterion::{criterion_group, criterion_main, Criterion};

const N_GEN: usize = 1_000_000;

fn unbounded_benchmark(c: &mut Criterion) {
    let mut generator = RandomGenerator::new_hardware(None).unwrap();
    c.bench_function("unbounded", |b| {
        b.iter(|| {
            (0..N_GEN).for_each(|_| {
                generator.generate_next();
            })
        })
    });
}

fn bounded_benchmark(c: &mut Criterion) {
    let mut generator = RandomGenerator::new_hardware(None).unwrap();
    let mut generator = generator
        .try_fork(1, N_GEN * 10_000)
        .unwrap()
        .next()
        .unwrap();
    c.bench_function("bounded", |b| {
        b.iter(|| {
            (0..N_GEN).for_each(|_| {
                generator.generate_next();
            })
        })
    });
}

criterion_group!(benches, unbounded_benchmark, bounded_benchmark);
criterion_main!(benches);
