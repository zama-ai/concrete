use crate::synthesizer::{
    SynthesizableCleartextVectorEntity, SynthesizableLweCiphertextEntity,
    SynthesizableLweCiphertextVectorEntity, SynthesizablePlaintextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{CleartextCount, LweCiphertextCount, LweDimension};
use concrete_core::specification::engines::LweCiphertextVectorDiscardingAffineTransformationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the discarding lwe affine transform operation.
pub fn bench<Engine, CiphertextVector, CleartextVector, Plaintext, OutputCiphertext>(
    c: &mut Criterion,
) where
    Engine: LweCiphertextVectorDiscardingAffineTransformationEngine<
        CiphertextVector,
        CleartextVector,
        Plaintext,
        OutputCiphertext,
    >,
    OutputCiphertext: SynthesizableLweCiphertextEntity,
    CiphertextVector:
        SynthesizableLweCiphertextVectorEntity<KeyDistribution = OutputCiphertext::KeyDistribution>,
    CleartextVector: SynthesizableCleartextVectorEntity,
    Plaintext: SynthesizablePlaintextEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextVectorDiscardingAffineTransformationEngine<
            CiphertextVector, 
            CleartextVector, 
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
                let (lwe_dim, ciphertext_count) = param.to_owned();
                let mut output = OutputCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                let inputs = CiphertextVector::synthesize(
                    &mut synthesizer,
                    lwe_dim,
                    ciphertext_count,
                    VARIANCE,
                );
                let weights = CleartextVector::synthesize(
                    &mut synthesizer,
                    CleartextCount(ciphertext_count.0),
                );
                let bias = Plaintext::synthesize(&mut synthesizer);
                b.iter(|| {
                    engine
                        .discard_affine_transform_lwe_ciphertext_vector(
                            black_box(&mut output),
                            black_box(&inputs),
                            black_box(&weights),
                            black_box(&bias),
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
const PARAMETERS: [(LweDimension, LweCiphertextCount); 6] = [
    (LweDimension(100), LweCiphertextCount(10)),
    (LweDimension(300), LweCiphertextCount(10)),
    (LweDimension(600), LweCiphertextCount(10)),
    (LweDimension(1000), LweCiphertextCount(10)),
    (LweDimension(3000), LweCiphertextCount(10)),
    (LweDimension(6000), LweCiphertextCount(10)),
];
