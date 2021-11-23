use crate::synthesizer::{
    SynthesizableLweCiphertextEntity, SynthesizableLweSecretKeyEntity,
    SynthesizablePlaintextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::specification::engines::LweCiphertextDecryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe decryption operation.
pub fn bench<Engine, SecretKey, Ciphertext, Plaintext>(c: &mut Criterion)
where
    Engine: LweCiphertextDecryptionEngine<SecretKey, Ciphertext, Plaintext>,
    SecretKey: SynthesizableLweSecretKeyEntity,
    Plaintext: SynthesizablePlaintextEntity,
    Ciphertext: SynthesizableLweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextDecryptionEngine<SecretKey, Ciphertext, Plaintext> for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let lwe_dim = param.to_owned();
                let secret_key = SecretKey::synthesize(&mut synthesizer, lwe_dim);
                let ciphertext = Ciphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                b.iter(|| {
                    black_box(
                        engine
                            .decrypt_lwe_ciphertext(black_box(&secret_key), black_box(&ciphertext))
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
const PARAMETERS: [LweDimension; 6] = [
    (LweDimension(100)),
    (LweDimension(300)),
    (LweDimension(600)),
    (LweDimension(1000)),
    (LweDimension(3000)),
    (LweDimension(6000)),
];
