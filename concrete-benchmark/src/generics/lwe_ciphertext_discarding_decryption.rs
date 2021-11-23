use crate::synthesizer::{
    SynthesizableLweCiphertextEntity, SynthesizableLweSecretKeyEntity,
    SynthesizablePlaintextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::specification::engines::LweCiphertextDiscardingDecryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the discarding lwe decryption operation.
pub fn bench<Engine, SecretKey, Ciphertext, Plaintext>(c: &mut Criterion)
where
    Engine: LweCiphertextDiscardingDecryptionEngine<SecretKey, Ciphertext, Plaintext>,
    SecretKey: SynthesizableLweSecretKeyEntity,
    Ciphertext: SynthesizableLweCiphertextEntity<KeyFlavor = SecretKey::KeyFlavor>,
    Plaintext: SynthesizablePlaintextEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextDiscardingDecryptionEngine<SecretKey, Ciphertext, Plaintext> for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let lwe_dim = param.to_owned();
                let mut plaintext = Plaintext::synthesize(&mut synthesizer);
                let secret_key = SecretKey::synthesize(&mut synthesizer, lwe_dim);
                let ciphertext = Ciphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                b.iter(|| {
                    engine
                        .discard_decrypt_lwe_ciphertext(
                            black_box(&secret_key),
                            black_box(&mut plaintext),
                            black_box(&ciphertext),
                        )
                        .unwrap();
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
