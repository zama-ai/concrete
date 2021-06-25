use concrete_commons::numeric::UnsignedInteger;
use concrete_core::math::random::{RandomGenerable, RandomGenerator, Uniform};
use concrete_core::math::tensor::Tensor;
use criterion::{black_box, Criterion};

pub fn bench<T: UnsignedInteger + RandomGenerable<Uniform>>(c: &mut Criterion) {
    let name = format!("random generate 100_000 u{}", T::BITS);
    let mut generator = RandomGenerator::new(None);
    c.bench_function(name.as_str(), |b| {
        b.iter(|| {
            let mut tensor: Tensor<Vec<T>> = Tensor::allocate(T::ZERO, black_box(100_000));
            black_box(generator.fill_tensor_with_random_uniform(&mut tensor));
        })
    });
}

pub fn bench_8(c: &mut Criterion) {
    bench::<u8>(c);
}

pub fn bench_16(c: &mut Criterion) {
    bench::<u16>(c);
}

pub fn bench_32(c: &mut Criterion) {
    bench::<u32>(c);
}

pub fn bench_64(c: &mut Criterion) {
    bench::<u64>(c);
}

pub fn bench_128(c: &mut Criterion) {
    bench::<u128>(c);
}
