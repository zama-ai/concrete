use crate::synthesizer::SynthesizableLweSecretKeyEntity;
use crate::utils::benchmark_name;
use concrete_commons::parameters::LweDimension;
use concrete_core::specification::engines::LweSecretKeyCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe secret key generation operation.
pub fn bench<Engine, SecretKey>(c: &mut Criterion)
where
    Engine: LweSecretKeyCreationEngine<SecretKey>,
    SecretKey: SynthesizableLweSecretKeyEntity,
{
    let mut group =
        c.benchmark_group(benchmark_name!(impl LweSecretKeyCreationEngine<SecretKey> for Engine));

    let mut engine = Engine::new().unwrap();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let lwe_dim = param.to_owned();
                b.iter(|| {
                    black_box(engine.create_lwe_secret_key(lwe_dim).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// The parameters the benchmark is executed against.
const PARAMETERS: [LweDimension; 6] = [
    (LweDimension(100)),
    (LweDimension(300)),
    (LweDimension(600)),
    (LweDimension(1000)),
    (LweDimension(3000)),
    (LweDimension(6000)),
];
