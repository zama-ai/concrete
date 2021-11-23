use crate::synthesizer::{
    SynthesizableGlweCiphertextEntity, SynthesizableLweBootstrapKeyEntity,
    SynthesizableLweCiphertextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_core::specification::engines::LweCiphertextDiscardingBootstrapEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe fusing bootstrap operation.
pub fn bench<Engine, BootstrapKey, Accumulator, InputCiphertext, OutputCiphertext>(
    c: &mut Criterion,
) where
    Engine: LweCiphertextDiscardingBootstrapEngine<
        BootstrapKey,
        Accumulator,
        InputCiphertext,
        OutputCiphertext,
    >,
    BootstrapKey: SynthesizableLweBootstrapKeyEntity,
    Accumulator:
        SynthesizableGlweCiphertextEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
    InputCiphertext:
        SynthesizableLweCiphertextEntity<KeyDistribution = BootstrapKey::InputKeyDistribution>,
    OutputCiphertext:
        SynthesizableLweCiphertextEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweCiphertextDiscardingBootstrapEngine<
            BootstrapKey, 
            Accumulator, 
            InputCiphertext, 
            OutputCiphertext
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (input_lwe_dim, glwe_dim, poly_size, base_log, level) = param.to_owned();
                let output_lwe_dim = LweDimension(glwe_dim.0 * poly_size.0);
                let bsk = BootstrapKey::synthesize(
                    &mut synthesizer,
                    input_lwe_dim,
                    poly_size,
                    glwe_dim,
                    base_log,
                    level,
                    VARIANCE,
                );
                let accumulator =
                    Accumulator::synthesize(&mut synthesizer, poly_size, glwe_dim, VARIANCE);
                let mut output_lwe =
                    OutputCiphertext::synthesize(&mut synthesizer, output_lwe_dim, VARIANCE);
                let input_lwe =
                    InputCiphertext::synthesize(&mut synthesizer, input_lwe_dim, VARIANCE);
                b.iter(|| {
                    engine
                        .discard_bootstrap_lwe_ciphertext(
                            black_box(&mut output_lwe),
                            black_box(&input_lwe),
                            black_box(&accumulator),
                            black_box(&bsk),
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
