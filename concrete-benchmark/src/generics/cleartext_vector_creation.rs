use crate::synthesizer::SynthesizableCleartextVectorEntity;
use crate::utils::{benchmark_name, IntegerValue};
use concrete_core::specification::engines::CleartextVectorCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the cleartext vector creation operation.
pub fn bench<Engine, Value, CleartextVector>(c: &mut Criterion)
where
    Engine: CleartextVectorCreationEngine<Value, CleartextVector>,
    Value: IntegerValue,
    CleartextVector: SynthesizableCleartextVectorEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl CleartextVectorCreationEngine<Value, CleartextVector> for Engine),
    );

    let mut engine = Engine::new().unwrap();

    group.bench_with_input(
        BenchmarkId::from_parameter("()".to_string()),
        &(),
        |b, _param| {
            b.iter(|| {
                black_box(
                    engine
                        .create_cleartext_vector(black_box(Value::any_vec(10).as_slice()))
                        .unwrap(),
                );
            });
        },
    );
    group.finish();
}
