use crate::synthesizer::{
    SynthesizableLweCiphertextEntity, SynthesizableLweKeyswitchKeyEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweDimension};
use concrete_core::specification::engines::LweCiphertextDiscardingKeyswitchEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the discarding lwe keyswitch operation.
pub fn bench<Engine, KeyswitchKey, InputCiphertext, OutputCiphertext>(c: &mut Criterion)
where
    Engine: LweCiphertextDiscardingKeyswitchEngine<KeyswitchKey, InputCiphertext, OutputCiphertext>,
    KeyswitchKey: SynthesizableLweKeyswitchKeyEntity,
    InputCiphertext:
        SynthesizableLweCiphertextEntity<KeyDistribution = KeyswitchKey::InputKeyDistribution>,
    OutputCiphertext:
        SynthesizableLweCiphertextEntity<KeyDistribution = KeyswitchKey::OutputKeyDistribution>,
{
    let mut group = c.benchmark_group(benchmark_name!(impl LweCiphertextDiscardingKeyswitchEngine<
            KeyswitchKey, 
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
                let (lwe_dim, base_log, level) = param.to_owned();
                let ksk = KeyswitchKey::synthesize(
                    &mut synthesizer,
                    lwe_dim,
                    lwe_dim,
                    base_log,
                    level,
                    VARIANCE,
                );
                let mut output_lwe =
                    OutputCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                let input_lwe = InputCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                b.iter(|| {
                    engine
                        .discard_keyswitch_lwe_ciphertext(
                            black_box(&mut output_lwe),
                            black_box(&input_lwe),
                            black_box(&ksk),
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
