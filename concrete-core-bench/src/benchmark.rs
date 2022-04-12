//! A module containing a specialized fixture extension for benchmarks.
//!
//! This module contains a [`BenchmarkFixture`] trait, which extends [`Fixture`] for all the types
//! implementing it, with two functions for benchmarking with criterion.
//!
//! Note :
//! ------
//!
//! Since `execute_engine` takes the pre-execution context by value (for good reasons), and returns
//! a post-execution context which must be properly disposed of, we can not rely on the usual timing
//! loop exposed by criterion. As a consequence, we rely on our own timing loop, which performs the
//! timing in batches.
//!
//! This means that all the pre-execution contexts for a batch are generated at once, out of the
//! timing loop, and stored in a buffer. The same goes for the post-execution contexts which are all
//! stored in a buffer and collected out of the timing loop.
//!
//! For this reason, it must be possible to store the whole batch of pre/post-execution contexts on
//! the benchmarked device. Depending on the available memory, this could be problematic. To
//! accomodate this situation, the input parameter `batch_size` can be used to provide a different
//! batch size.
use concrete_core::prelude::AbstractEngine;
use concrete_core_fixture::fixture::Fixture;
use concrete_core_fixture::generation::{IntegerPrecision, Maker};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion};
use std::cmp::min;
use std::ops::Add;
use std::time::{Duration, Instant};

/// Default value for the batch size.
pub const DEFAULT_BATCH_SIZE: u64 = 1000;

/// An extension to the `Fixture` trait for benchmarking with criterion.
pub trait BenchmarkFixture<Precision: IntegerPrecision, Engine: AbstractEngine, RelatedEntities>:
    Fixture<Precision, Engine, RelatedEntities>
{
    /// Benchmarks all the parameters defined by the fixture.
    fn bench_all_parameters(
        maker: &mut Maker,
        engine: &mut Engine,
        criterion: &mut Criterion,
        batch_size: Option<u64>,
    ) {
        let mut group = criterion.benchmark_group(format!(
            "{}<{}, {}, {}>",
            type_name::<Self>(),
            type_name::<Precision>(),
            type_name::<Engine>(),
            type_name::<RelatedEntities>()
        ));
        for params in Self::generate_parameters_iterator() {
            Self::bench(maker, engine, params, &mut group, batch_size);
        }
        group.finish();
    }

    /// Benchmark one parameter in the fixture benchmark group.
    fn bench(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: Self::Parameters,
        fixture_group: &mut BenchmarkGroup<WallTime>,
        batch_size: Option<u64>,
    ) {
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);

        // We generate the prototypes once (used for all repetitions of the benchmark).
        let repetition_proto = Self::generate_random_repetition_prototypes(&parameters, maker);
        let sample_proto =
            Self::generate_random_sample_prototypes(&parameters, maker, &repetition_proto);

        // The benchmark itself does not use the standard iterators of `criterion`, because the
        // pre-context is passed by value, and the post-context must be gathered to be
        // properly disposed afterward.
        fixture_group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", parameters)),
            &(repetition_proto, sample_proto),
            |b, (repetition_proto, sample_proto)| {
                b.iter_custom(|iterations_to_go| {
                    // We initialize the vectors which will store the input and output values of the
                    // batch.
                    let mut inputs_vec: Vec<Self::PreExecutionContext> =
                        Vec::with_capacity(batch_size as usize);
                    let mut outputs_vec: Vec<Self::PostExecutionContext> =
                        Vec::with_capacity(batch_size as usize);

                    // We iterate until we reached the number of iterations to be performed.
                    let mut iterations_counter = 0;
                    let mut duration = Duration::new(0, 0);
                    while iterations_counter < iterations_to_go {
                        let batch_size = min(batch_size, iterations_to_go - iterations_counter);
                        iterations_counter += batch_size;

                        (0..batch_size).for_each(|_| {
                            inputs_vec.push(Self::prepare_context(
                                &parameters,
                                maker,
                                repetition_proto,
                                sample_proto,
                            ))
                        });

                        let start = Instant::now();
                        inputs_vec.drain(..).for_each(|input| {
                            outputs_vec.push(Self::execute_engine(&parameters, engine, input));
                        });
                        duration = duration.add(start.elapsed());

                        outputs_vec.drain(..).for_each(|output| {
                            Self::process_context(
                                &parameters,
                                maker,
                                repetition_proto,
                                sample_proto,
                                output,
                            );
                        })
                    }

                    duration
                })
            },
        );
    }
}

impl<Precision, Engine, RelatedEntities, Fix> BenchmarkFixture<Precision, Engine, RelatedEntities>
    for Fix
where
    Fix: Fixture<Precision, Engine, RelatedEntities>,
    Precision: IntegerPrecision,
    Engine: AbstractEngine,
{
}

/// A function returning the name of a type (just the name, not the path).
fn type_name<T: ?Sized>() -> &'static str {
    std::any::type_name::<T>()
        .split("::")
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}
