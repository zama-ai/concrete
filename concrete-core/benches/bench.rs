use criterion::{criterion_group, criterion_main};

mod bootstrap;
mod decomposition;
mod keyswitch;
mod random;

criterion_group!(bootstrap_b, bootstrap::bench_32, bootstrap::bench_64);
criterion_group!(keyswitch_b, keyswitch::bench_32, keyswitch::bench_64);
criterion_group!(
    random_b,
    random::bench_8,
    random::bench_16,
    random::bench_32,
    random::bench_64,
    random::bench_128
);
criterion_group!(
    decomposition_b,
    decomposition::bench_32,
    decomposition::bench_64
);

criterion_main!(bootstrap_b, keyswitch_b, random_b, decomposition_b);
