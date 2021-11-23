use crate::synthesizer::{
    SynthesizableLweCiphertextVectorEntity, SynthesizableLweSecretKeyEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
use concrete_core::specification::engines::LweCiphertextVectorZeroEncryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe vector zero encryption operation.
pub fn bench<Engine, SecretKey, CiphertextVector>(c: &mut Criterion)
where
    Engine: LweCiphertextVectorZeroEncryptionEngine<SecretKey, CiphertextVector>,
    SecretKey: SynthesizableLweSecretKeyEntity,
    CiphertextVector:
        SynthesizableLweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextVectorZeroEncryptionEngine<
            SecretKey,
            CiphertextVector
            > for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (lwe_dim, ciphertext_count) = param.to_owned();
                let secret_key = SecretKey::synthesize(&mut synthesizer, lwe_dim);
                b.iter(|| {
                    black_box(
                        engine
                            .zero_encrypt_lwe_ciphertext_vector(
                                black_box(&secret_key),
                                black_box(VARIANCE),
                                black_box(ciphertext_count),
                            )
                            .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

/// The variance used to encrypt everything in the benchmark.
const VARIANCE: Variance = Variance(0.00000001);

/// The parameters the benchmark is executed against.
const PARAMETERS: [(LweDimension, LweCiphertextCount); 6] = [
    (LweDimension(100), LweCiphertextCount(100)),
    (LweDimension(300), LweCiphertextCount(100)),
    (LweDimension(600), LweCiphertextCount(100)),
    (LweDimension(1000), LweCiphertextCount(100)),
    (LweDimension(3000), LweCiphertextCount(100)),
    (LweDimension(6000), LweCiphertextCount(100)),
];
