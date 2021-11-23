use crate::synthesizer::{
    SynthesizableGlweCiphertextVectorEntity, SynthesizableGlweSecretKeyEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};
use concrete_core::specification::engines::GlweCiphertextVectorZeroEncryptionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the glwe vector zero encryption operation.
pub fn bench<Engine, SecretKey, CiphertextVector>(c: &mut Criterion)
where
    Engine: GlweCiphertextVectorZeroEncryptionEngine<SecretKey, CiphertextVector>,
    SecretKey: SynthesizableGlweSecretKeyEntity,
    CiphertextVector: SynthesizableGlweCiphertextVectorEntity<KeyFlavor = SecretKey::KeyFlavor>,
{
    let mut group = c.benchmark_group(benchmark_name!(
        impl GlweCiphertextVectorZeroEncryptionEngine<SecretKey, CiphertextVector> for Engine
    ));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (glwe_dim, poly_size, ciphertext_count) = param.to_owned();
                let secret_key = SecretKey::synthesize(&mut synthesizer, poly_size, glwe_dim);
                b.iter(|| {
                    black_box(
                        engine
                            .zero_encrypt_glwe_ciphertext_vector(
                                black_box(&secret_key),
                                VARIANCE,
                                ciphertext_count,
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
const PARAMETERS: [(GlweDimension, PolynomialSize, GlweCiphertextCount); 5] = [
    (
        GlweDimension(1),
        PolynomialSize(256),
        GlweCiphertextCount(100),
    ),
    (
        GlweDimension(1),
        PolynomialSize(512),
        GlweCiphertextCount(100),
    ),
    (
        GlweDimension(1),
        PolynomialSize(1024),
        GlweCiphertextCount(100),
    ),
    (
        GlweDimension(1),
        PolynomialSize(2048),
        GlweCiphertextCount(100),
    ),
    (
        GlweDimension(1),
        PolynomialSize(4096),
        GlweCiphertextCount(100),
    ),
];
