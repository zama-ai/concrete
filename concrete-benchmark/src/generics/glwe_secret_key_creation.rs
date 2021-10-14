use crate::synthesizer::SynthesizableGlweSecretKeyEntity;
use crate::utils::benchmark_name;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::specification::engines::GlweSecretKeyCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the glwe secret key generation operation.
pub fn bench<Engine, SecretKey>(c: &mut Criterion)
where
    Engine: GlweSecretKeyCreationEngine<SecretKey>,
    SecretKey: SynthesizableGlweSecretKeyEntity,
{
    let mut group =
        c.benchmark_group(benchmark_name!(impl GlweSecretKeyCreationEngine<SecretKey> for Engine));
    let mut engine = Engine::new().unwrap();
    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (glwe_dim, poly_size) = param.to_owned();

                b.iter(|| {
                    black_box(engine.create_glwe_secret_key(glwe_dim, poly_size).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// The parameters the benchmark is executed against.
const PARAMETERS: [(GlweDimension, PolynomialSize); 5] = [
    (GlweDimension(1), PolynomialSize(256)),
    (GlweDimension(1), PolynomialSize(512)),
    (GlweDimension(1), PolynomialSize(1024)),
    (GlweDimension(1), PolynomialSize(2048)),
    (GlweDimension(1), PolynomialSize(4096)),
];
