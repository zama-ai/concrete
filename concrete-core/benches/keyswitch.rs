use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::crypto::lwe::{LweCiphertext, LweKeyswitchKey};
use concrete_core::crypto::LweDimension;
use concrete_core::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::math::random::{RandomGenerable, UniformMsb};
use concrete_core::math::torus::UnsignedTorus;
use criterion::{BenchmarkId, Criterion};
use itertools::iproduct;

pub fn bench<T: UnsignedTorus + RandomGenerable<UniformMsb>>(c: &mut Criterion) {
    // fix a set of parameters
    let base_log_level = vec![(4, 3), (6, 2), (10, 1)]; // a parameter of the gadget matrix
    let dimensions_before = vec![1024];
    let dimensions_after = vec![512];

    let params = iproduct!(base_log_level, dimensions_before, dimensions_after);

    let mut group = c.benchmark_group("compilo-keyswitch");
    for p in params {
        // group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "p={}-bg={}-l={}-nin={}-nout={}",
                T::BITS,
                (p.0).0,
                (p.0).1,
                p.1,
                p.2
            )),
            &p,
            |b, p| {
                let base_log = DecompositionBaseLog((p.0).0);
                let level = DecompositionLevelCount((p.0).1);
                let dimension_before = LweDimension(p.1);
                let dimension_after = LweDimension(p.2);

                // create the before ciphertexts and the after ciphertexts
                let ciphertexts_before =
                    LweCiphertext::allocate(T::ZERO, dimension_before.to_lwe_size());
                let mut ciphertexts_after =
                    LweCiphertext::allocate(T::ZERO, dimension_after.to_lwe_size());

                // key switching key generation
                let ksk = LweKeyswitchKey::allocate(
                    T::ZERO,
                    level,
                    base_log,
                    dimension_before,
                    dimension_after,
                );

                b.iter(|| {
                    // key switch before -> after
                    ksk.keyswitch_ciphertext(&mut ciphertexts_after, &ciphertexts_before);
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
