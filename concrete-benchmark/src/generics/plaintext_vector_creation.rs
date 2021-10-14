use crate::synthesizer::SynthesizablePlaintextVectorEntity;
use crate::utils::{benchmark_name, IntegerValue};
use concrete_core::specification::engines::PlaintextVectorCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the plaintext vector creation operation.
pub fn bench<Engine, Value, PlaintextVector>(c: &mut Criterion)
where
    Engine: PlaintextVectorCreationEngine<Value, PlaintextVector>,
    Value: IntegerValue,
    PlaintextVector: SynthesizablePlaintextVectorEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl PlaintextVectorCreationEngine<Value, PlaintextVector> for Engine),
    );

    let mut engine = Engine::new().unwrap();

    group.bench_with_input(
        BenchmarkId::from_parameter("()".to_string()),
        &(),
        |b, _param| {
            b.iter(|| {
                black_box(
                    engine
                        .create_plaintext_vector(black_box(Value::any_vec(10).as_slice()))
                        .unwrap(),
                );
            });
        },
    );
    group.finish();
}
