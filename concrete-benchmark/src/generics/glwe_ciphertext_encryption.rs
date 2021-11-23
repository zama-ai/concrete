use crate::synthesizer::{
    SynthesizableGlweCiphertextEntity, SynthesizableGlweSecretKeyEntity,
    SynthesizablePlaintextVectorEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, PlaintextCount, PolynomialSize};
use concrete_core::specification::engines::GlweCiphertextEncryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the glwe encryption operation.
pub fn bench<Engine, SecretKey, PlaintextVector, Ciphertext>(c: &mut Criterion)
where
    Engine: GlweCiphertextEncryptionEngine<SecretKey, PlaintextVector, Ciphertext>,
    SecretKey: SynthesizableGlweSecretKeyEntity,
    PlaintextVector: SynthesizablePlaintextVectorEntity,
    Ciphertext: SynthesizableGlweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    let mut group = c.benchmark_group(benchmark_name!(impl GlweCiphertextEncryptionEngine<
            SecretKey, 
            PlaintextVector, 
            Ciphertext
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (glwe_dimension, poly_size) = param.to_owned();
                let plaintext_vector =
                    PlaintextVector::synthesize(&mut synthesizer, PlaintextCount(poly_size.0));
                let secret_key = SecretKey::synthesize(&mut synthesizer, poly_size, glwe_dimension);
                b.iter(|| {
                    black_box(
                        engine
                            .encrypt_glwe_ciphertext(
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
const PARAMETERS: [(GlweDimension, PolynomialSize); 5] = [
    (GlweDimension(1), PolynomialSize(256)),
    (GlweDimension(1), PolynomialSize(512)),
    (GlweDimension(1), PolynomialSize(1024)),
    (GlweDimension(1), PolynomialSize(2048)),
    (GlweDimension(1), PolynomialSize(4096)),
];
