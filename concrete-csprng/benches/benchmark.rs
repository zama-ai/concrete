use concrete_csprng::RandomGenerator;
use criterion::{criterion_group, criterion_main, Criterion};

const N_GEN: usize = 1_000_000;

fn unforked_benchmark(c: &mut Criterion) {
    let mut generator = RandomGenerator::new_hardware(None).unwrap();
    c.bench_function("unforked", |b| {
        b.iter(|| {
            (0..N_GEN).for_each(|_| {
                generator.generate_next();
            })
        })
    });
}

fn sequential_forked_benchmark(c: &mut Criterion) {
    let mut generator = RandomGenerator::new_hardware(None).unwrap();
    let mut generator = generator
        .try_sequential_fork(1, N_GEN * 10_000)
        .unwrap()
        .next()
        .unwrap();
    c.bench_function("sequential_forked", |b| {
        b.iter(|| {
            (0..N_GEN).for_each(|_| {
                generator.generate_next();
            })
        })
    });
}

fn alternate_forked_benchmark(c: &mut Criterion) {
    let mut generator = RandomGenerator::new_hardware(None).unwrap();
    let mut generator = generator.try_alternate_fork(1).unwrap().next().unwrap();
    c.bench_function("alternate_forked", |b| {
        b.iter(|| {
            (0..N_GEN).for_each(|_| {
                generator.generate_next();
            })
        })
    });
}

criterion_group!(
    benches,
    unforked_benchmark,
    alternate_forked_benchmark,
    sequential_forked_benchmark
);
criterion_main!(benches);
