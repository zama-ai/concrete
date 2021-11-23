use crate::synthesizer::{SynthesizableLweBootstrapKeyEntity, Synthesizer};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::specification::engines::LweBootstrapKeyConversionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe bootstrap key conversion operation.
pub fn bench<Engine, InputBootstrapKey, OutputBootstrapKey>(c: &mut Criterion)
where
    Engine: LweBootstrapKeyConversionEngine<InputBootstrapKey, OutputBootstrapKey>,
    InputBootstrapKey: SynthesizableLweBootstrapKeyEntity,
    OutputBootstrapKey: SynthesizableLweBootstrapKeyEntity<
        InputKeyDistribution = InputBootstrapKey::InputKeyDistribution,
        OutputKeyDistribution = InputBootstrapKey::OutputKeyDistribution,
    >,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweBootstrapKeyConversionEngine<
            InputBootstrapKey, 
            OutputBootstrapKey
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (input_lwe_dim, glwe_dim, poly_size, base_log, level) = param.to_owned();
                let bsk: InputBootstrapKey = InputBootstrapKey::synthesize(
                    &mut synthesizer,
                    input_lwe_dim,
                    poly_size,
                    glwe_dim,
                    base_log,
                    level,
                    VARIANCE,
                );
                b.iter(|| {
                    black_box(engine.convert_lwe_bootstrap_key(&bsk).unwrap());
                });
            },
        );
    }
    group.finish();
}

/// The variance used to encrypt everything in the benchmark.
const VARIANCE: Variance = Variance(0.00000001);

/// The parameters the benchmark is executed against.
const PARAMETERS: [(
    LweDimension,
    GlweDimension,
    PolynomialSize,
    DecompositionBaseLog,
    DecompositionLevelCount,
); 5] = [
    (
        LweDimension(100),
        GlweDimension(1),
        PolynomialSize(256),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(100),
        GlweDimension(1),
        PolynomialSize(512),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(100),
        GlweDimension(1),
        PolynomialSize(1024),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(100),
        GlweDimension(1),
        PolynomialSize(2048),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(100),
        GlweDimension(1),
        PolynomialSize(4096),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
];
