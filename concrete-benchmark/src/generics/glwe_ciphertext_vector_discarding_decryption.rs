use crate::synthesizer::{
    SynthesizableGlweCiphertextVectorEntity, SynthesizableGlweSecretKeyEntity,
    SynthesizablePlaintextVectorEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    GlweCiphertextCount, GlweDimension, PlaintextCount, PolynomialSize,
};
use concrete_core::specification::engines::GlweCiphertextVectorDiscardingDecryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the glwe vector discarding decryption operation.
pub fn bench<Engine, SecretKey, CiphertextVector, PlaintextVector>(c: &mut Criterion)
where
    Engine: GlweCiphertextVectorDiscardingDecryptionEngine<
        SecretKey,
        CiphertextVector,
        PlaintextVector,
    >,
    SecretKey: SynthesizableGlweSecretKeyEntity,
    CiphertextVector:
        SynthesizableGlweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
    PlaintextVector: SynthesizablePlaintextVectorEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl GlweCiphertextVectorDiscardingDecryptionEngine<
            SecretKey, 
            CiphertextVector,
            PlaintextVector
            > for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (glwe_dim, poly_size, ciphertext_count) = param.to_owned();
                let mut plaintext_vector = PlaintextVector::synthesize(
                    &mut synthesizer,
                    PlaintextCount(poly_size.0 * ciphertext_count.0),
                );
                let secret_key = SecretKey::synthesize(&mut synthesizer, poly_size, glwe_dim);
                let ciphertext_vector = CiphertextVector::synthesize(
                    &mut synthesizer,
                    poly_size,
                    glwe_dim,
                    ciphertext_count,
                    VARIANCE,
                );
                b.iter(|| {
                    engine
                        .discard_decrypt_glwe_ciphertext_vector(
                            black_box(&secret_key),
                            black_box(&mut plaintext_vector),
                            black_box(&ciphertext_vector),
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
const PARAMETERS: [(GlweDimension, PolynomialSize, GlweCiphertextCount); 5] = [
    (
        GlweDimension(1),
        PolynomialSize(256),
        GlweCiphertextCount(1),
    ),
    (
        GlweDimension(1),
        PolynomialSize(512),
        GlweCiphertextCount(1),
    ),
    (
        GlweDimension(1),
        PolynomialSize(1024),
        GlweCiphertextCount(1),
    ),
    (
        GlweDimension(1),
        PolynomialSize(2048),
        GlweCiphertextCount(1),
    ),
    (
        GlweDimension(1),
        PolynomialSize(4096),
        GlweCiphertextCount(1),
    ),
];
