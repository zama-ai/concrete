use crate::synthesizer::SynthesizableCleartextEntity;
use crate::utils::{benchmark_name, IntegerValue};
use concrete_core::specification::engines::CleartextCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the cleartext creation operation.
pub fn bench<Engine, Value, Cleartext>(c: &mut Criterion)
where
    Engine: CleartextCreationEngine<Value, Cleartext>,
    Value: IntegerValue,
    Cleartext: SynthesizableCleartextEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl CleartextCreationEngine<Value, Cleartext> for Engine),
    );

    let mut engine = Engine::new().unwrap();

    group.bench_with_input(
        BenchmarkId::from_parameter("()".to_string()),
        &(),
        |b, _param| {
            b.iter(|| {
                black_box(engine.create_cleartext(black_box(&Value::any())).unwrap());
            });
        },
    );
    group.finish();
}
