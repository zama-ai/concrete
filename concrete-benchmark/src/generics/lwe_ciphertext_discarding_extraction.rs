use crate::synthesizer::{
    SynthesizableGlweCiphertextEntity, SynthesizableLweCiphertextEntity, Synthesizer,
};
use crate::utils::benchmark_name;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{GlweDimension, LweDimension, MonomialIndex, PolynomialSize};
use concrete_core::specification::engines::LweCiphertextDiscardingExtractionEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the discarding lwe sample extraction.
pub fn bench<Engine, GlweCiphertext, LweCiphertext>(c: &mut Criterion)
where
    Engine: LweCiphertextDiscardingExtractionEngine<GlweCiphertext, LweCiphertext>,
    GlweCiphertext: SynthesizableGlweCiphertextEntity,
    LweCiphertext:
        SynthesizableLweCiphertextEntity<KeyDistribution = GlweCiphertext::KeyDistribution>,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl LweCiphertextDiscardingExtractionEngine<
            GlweCiphertext, 
            LweCiphertext
            > for Engine),
    );

    let mut engine = Engine::new().unwrap();
    let mut synthesizer = Synthesizer::default();

    for param in PARAMETERS {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", param)),
            &param,
            |b, param| {
                let (glwe_dim, poly_size) = param.to_owned();
                let lwe_dim = LweDimension(glwe_dim.0 * poly_size.0);
                let mut lwe_ciphertext =
                    LweCiphertext::synthesize(&mut synthesizer, lwe_dim, VARIANCE);
                let glwe_ciphertext =
                    GlweCiphertext::synthesize(&mut synthesizer, poly_size, glwe_dim, VARIANCE);
                b.iter(|| {
                    engine
                        .discard_extract_lwe_ciphertext(
                            black_box(&mut lwe_ciphertext),
                            black_box(&glwe_ciphertext),
                            black_box(MonomialIndex(0)),
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
const PARAMETERS: [(GlweDimension, PolynomialSize); 10] = [
    (GlweDimension(1), PolynomialSize(256)),
    (GlweDimension(1), PolynomialSize(512)),
    (GlweDimension(1), PolynomialSize(1024)),
    (GlweDimension(1), PolynomialSize(2048)),
    (GlweDimension(1), PolynomialSize(4096)),
    (GlweDimension(3), PolynomialSize(256)),
    (GlweDimension(3), PolynomialSize(512)),
    (GlweDimension(3), PolynomialSize(1024)),
    (GlweDimension(3), PolynomialSize(2048)),
    (GlweDimension(3), PolynomialSize(4096)),
];
