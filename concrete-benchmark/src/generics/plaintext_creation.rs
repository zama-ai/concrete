use crate::synthesizer::SynthesizablePlaintextEntity;
use crate::utils::{benchmark_name, IntegerValue};
use concrete_core::specification::engines::PlaintextCreationEngine;
use criterion::{black_box, BenchmarkId, Criterion};

/// A generic function benchmarking the plaintext creation operation.
pub fn bench<Engine, Value, Plaintext>(c: &mut Criterion)
where
    Engine: PlaintextCreationEngine<Value, Plaintext>,
    Value: IntegerValue,
    Plaintext: SynthesizablePlaintextEntity,
{
    let mut group = c.benchmark_group(
        benchmark_name!(impl PlaintextCreationEngine<Value, Plaintext> for Engine),
    );

    let mut engine = Engine::new().unwrap();

    group.bench_with_input(
        BenchmarkId::from_parameter("()".to_string()),
        &(),
        |b, _param| {
            b.iter(|| {
                black_box(engine.create_plaintext(black_box(&Value::any())).unwrap());
            });
        },
    );
    group.finish();
}
