use crate::synthesizer::{
    SynthesizableLweCiphertextEntity, SynthesizablePlaintextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::specification::engines::LweCiphertextPlaintextDiscardingAdditionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the discarding lwe plaintext addition operation.
pub fn bench<Engine, InputCiphertext, Plaintext, OutputCiphertext>(c: &mut Criterion)
where
    Engine: LweCiphertextPlaintextDiscardingAdditionEngine<
        InputCiphertext,
        Plaintext,
        OutputCiphertext,
    >,
    InputCiphertext: SynthesizableLweCiphertextEntity,
    OutputCiphertext: SynthesizableLweCiphertextEntity<KeyFlavor = InputCiphertext::KeyFlavor>,
    Plaintext: SynthesizablePlaintextEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextPlaintextDiscardingAdditionEngine<
            InputCiphertext, 
            Plaintext, 
            OutputCiphertext
        > for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let lwe_dim = param.to_owned();
                let mut output = OutputCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                let input_1 = InputCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                let input_2 = Plaintext::synthesize(&mut synthesizer);
                b.iter(|| {
                    engine
                        .discard_add_lwe_ciphertext_plaintext(
                            black_box(&mut output),
                            black_box(&input_1),
                            black_box(&input_2),
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
