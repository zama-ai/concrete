//! A module containing a generic fixture trait for `concrete-core` operators.
//!
//! The central abstraction of the library is the [`Fixture`] trait. This trait defines the logic
//! used to sample / test / benchmark the implementors of any of the `*Engine` traits defined in
//! `concrete-core`. This logic is always the same and, depending on the use of the fixture, boils
//! down to:
//!
//! + Iterating over a set of parameters multiple times
//! + Generating random raw inputs (where _raw_ is a type derived of rust´s integers like u32, u64,
//! Vec<u32>,   etc...)
//! + Preparing the operator pre-execution context using the raw inputs
//! + Executing the operator in this context
//! + Collecting the post-execution context, cleaning the entities, and extracting raw outputs
//! + Predicting the statistical properties of the raw output sample using the raw inputs
//! + Check that the output sample properties match the predictions
//!
//! For any given `*Engine` trait, a matching `*Fixture` type is defined, which _generically_
//! implements [`Fixture`]. Here _generically_ means that the implementation of the [`Fixture`]
//! trait does not fix the `Engine` and `Precision` type parameters, but rather restricts them to a
//! family of types. In particular, the `Engine` generic type parameter must implement the `*Engine`
//! trait in question. This `*Fixture` type can then be used to sample / test / benchmark any
//! implementor of the `*Engine` trait.
//!
//! In particular, once the [`Fixture`] mandatory methods and types are defined, the user can
//! benefit from the default methods [`Fixture::sample`], [`Fixture::test`] or [`Fixture::stress`].
use crate::generation::{IntegerPrecision, Maker};
use crate::{Repetitions, SampleSize};
use concrete_core::prelude::AbstractEngine;
use std::ops::BitAnd;

/// A trait for types implementing a fixture for a particular engine trait.
///
/// To understand how the different pieces fit, see how the default methods `sample`, `test`,
/// `stress` and `stress_all` use the associated types and methods.
pub trait Fixture<Precision: IntegerPrecision, Engine: AbstractEngine, RelatedEntities> {
    /// A type containing the parameters needed to generate input data.
    type Parameters;

    /// A type containing the raw inputs used to generate the execution context.
    type RawInputs;

    /// A type containing all the objects which must exist for the engine to be executed.
    type PreExecutionContext;

    /// A type containing prototypical secret keys passed around the engine execution.
    type SecretKeyPrototypes;

    /// A type containing all the objects existing after the engine got executed.
    type PostExecutionContext;

    /// A type containing the (eventually decrypted) raw outputs of the operator execution.
    type RawOutputs;

    /// A type containing the prediction for the characteristics of the output sample distribution.
    type Prediction;

    /// A method which outputs an iterator over parameters.
    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>>;

    /// A method which generates random raw inputs.
    fn generate_random_raw_inputs(parameters: &Self::Parameters) -> Self::RawInputs;

    /// A method which prepares the pre-execution context for the engine execution.
    ///
    /// The first returned value, can be used to store prototypical secret keys used in the
    /// `process_context` method to decrypt values in the output context.
    fn prepare_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        raw_inputs: &Self::RawInputs,
    ) -> (Self::SecretKeyPrototypes, Self::PreExecutionContext);

    /// A method which executes the engine in the pre-execution context, returning the
    /// post-execution context.
    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext;

    /// A method which processes the post-execution context, and returns the raw outputs.
    fn process_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        secret_keys: Self::SecretKeyPrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::RawOutputs;

    /// A method which computes the prediction for the statistics of the raw output sample.
    fn compute_prediction(
        parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> Self::Prediction;

    /// A method which verify that the predictions are met by the raw output sample.
    fn check_prediction(
        parameters: &Self::Parameters,
        prediction: &Self::Prediction,
        actual: &[Self::RawOutputs],
    ) -> bool;

    /// A method which verifies the statistical properties of a sample of engine executions, over
    /// multiple randomly generated raw inputs, over multiple sets of parameters.
    fn stress_all_parameters(
        maker: &mut Maker,
        engine: &mut Engine,
        repetitions: Repetitions,
        sample_size: SampleSize,
    ) -> bool {
        Self::generate_parameters_iterator()
            .map(|param| Self::stress(maker, engine, &param, repetitions, sample_size))
            .reduce(BitAnd::bitand)
            .unwrap()
    }

    /// A method which verifies the statistical properties of a sample of engine executions, over
    /// multiple randomly generated raw inputs, for a fixed set of parameters.
    fn stress(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: &Self::Parameters,
        repetitions: Repetitions,
        sample_size: SampleSize,
    ) -> bool {
        (0..repetitions.0)
            .map(|_| {
                Self::test(
                    maker,
                    engine,
                    parameters,
                    &Self::generate_random_raw_inputs(parameters),
                    sample_size,
                )
            })
            .reduce(BitAnd::bitand)
            .expect("At least one repetition is needed.")
    }

    /// A method which verifies the statistical properties of a sample of engine execution, for a
    /// fixed set of raw inputs and a fixed set of parameters.
    fn test(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> bool {
        let prediction = Self::compute_prediction(parameters, raw_inputs, sample_size);
        let outputs = Self::sample(maker, engine, parameters, raw_inputs, sample_size);
        Self::check_prediction(parameters, &prediction, outputs.as_slice())
    }

    /// A method which generates a sample of engine execution, for a fixed set of raw inputs and a
    /// fixed set of parameters.
    fn sample(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: &Self::Parameters,
        raw_inputs: &Self::RawInputs,
        sample_size: SampleSize,
    ) -> Vec<Self::RawOutputs> {
        let mut outputs = Vec::with_capacity(sample_size.0);
        for _ in 0..sample_size.0 {
            let (keys, inputs) = Self::prepare_context(parameters, maker, raw_inputs);
            let out = Self::execute_engine(parameters, engine, inputs);
            outputs.push(Self::process_context(parameters, maker, keys, out));
        }
        outputs
    }
}

mod cleartext_creation;

pub use cleartext_creation::*;

mod lwe_ciphertext_encryption;

pub use lwe_ciphertext_encryption::*;
