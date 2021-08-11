use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::math::decomposition::SignedDecomposer;
use concrete_core::math::tensor::Tensor;
use criterion::{black_box, BenchmarkId, Criterion};
use itertools::iproduct;

pub fn bench<T: UnsignedInteger>(c: &mut Criterion) {
    // fix a set of parameters
    let base_log_level = vec![(4, 3), (6, 2), (10, 2)];
    let tensor_length = vec![100_000, 20_000_000];
    let params = iproduct!(base_log_level, tensor_length);

    let mut group = c.benchmark_group("decomposition");
    for p in params {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "p={}-bg={}-l={}-length={}",
                T::BITS,
                (p.0).0,
                (p.0).1,
                p.1,
            )),
            &p,
            |b, p| {
                let base_log = DecompositionBaseLog((p.0).0);
                let level = DecompositionLevelCount((p.0).1);
                let length = p.1;

                let decomposer = SignedDecomposer::new(base_log, level);
                let tensor = Tensor::allocate(T::ZERO, length);

                b.iter(|| {
                    let mut decomp = decomposer.decompose_tensor(&tensor);
                    while let Some(val) = decomp.next_term() {
                        black_box(val);
                    }
                });
            },
        );
    }
    group.finish();
}

pub fn bench_32(c: &mut Criterion) {
    bench::<u32>(c);
}

pub fn bench_64(c: &mut Criterion) {
    bench::<u64>(c);
}
