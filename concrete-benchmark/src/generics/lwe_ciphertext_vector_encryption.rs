use crate::synthesizer::{
    SynthesizableLweCiphertextVectorEntity, SynthesizableLweSecretKeyEntity,
    SynthesizablePlaintextVectorEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{LweCiphertextCount, LweDimension, PlaintextCount};
use concrete_core::specification::engines::LweCiphertextVectorEncryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe vector encryption operation.
pub fn bench<Engine, SecretKey, PlaintextVector, CiphertextVector>(c: &mut Criterion)
where
    Engine: LweCiphertextVectorEncryptionEngine<SecretKey, PlaintextVector, CiphertextVector>,
    SecretKey: SynthesizableLweSecretKeyEntity,
    PlaintextVector: SynthesizablePlaintextVectorEntity,
    CiphertextVector:
        SynthesizableLweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweCiphertextVectorEncryptionEngine<
            SecretKey, 
            PlaintextVector, 
            CiphertextVector
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (lwe_dim, ciphertext_count) = param.to_owned();
                let plaintext_vector = PlaintextVector::synthesize(
                    &mut synthesizer,
                    PlaintextCount(ciphertext_count.0),
                );
                let secret_key = SecretKey::synthesize(&mut synthesizer, lwe_dim);
                b.iter(|| {
                    black_box(
                        engine
                            .encrypt_lwe_ciphertext_vector(
                                black_box(&secret_key),
                                black_box(&plaintext_vector),
                                black_box(VARIANCE),
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
