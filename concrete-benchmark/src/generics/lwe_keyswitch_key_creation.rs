use crate::synthesizer::{
    SynthesizableLweKeyswitchKeyEntity, SynthesizableLweSecretKeyEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::specification::engines::LweKeyswitchKeyCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the lwe keyswitch key generation operation.
pub fn bench<Engine, InputSecretKey, OutputSecretKey, KeyswitchKey>(c: &mut Criterion)
where
    Engine: LweKeyswitchKeyCreationEngine<InputSecretKey, OutputSecretKey, KeyswitchKey>,
    InputSecretKey: SynthesizableLweSecretKeyEntity,
    OutputSecretKey: SynthesizableLweSecretKeyEntity,
    KeyswitchKey: SynthesizableLweKeyswitchKeyEntity<
        InputKeyDistribution = InputSecretKey::KeyDistribution,
        OutputKeyDistribution = OutputSecretKey::KeyDistribution,
    >,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweKeyswitchKeyCreationEngine<
            InputSecretKey, 
            OutputSecretKey, 
            KeyswitchKey
            > for Engine));

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (lwe_dim, base_log, level) = param.to_owned();
                let input_lwe_sk = InputSecretKey::synthesize(&mut synthesizer, lwe_dim);
                let output_lwe_sk = OutputSecretKey::synthesize(&mut synthesizer, lwe_dim);
                b.iter(|| {
                    black_box(
                        engine
                            .create_lwe_keyswitch_key(
                                &input_lwe_sk,
                                &output_lwe_sk,
                                level,
                                base_log,
                                VARIANCE,
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
const PARAMETERS: [(LweDimension, DecompositionBaseLog, DecompositionLevelCount); 5] = [
    (
        LweDimension(100),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(200),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(300),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(400),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
    (
        LweDimension(500),
        DecompositionBaseLog(2),
        DecompositionLevelCount(3),
    ),
];
