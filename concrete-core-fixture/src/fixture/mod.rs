//! A module containing a generic fixture trait for `concrete-core` operators.
//!
//! The central abstraction of the library is the [`Fixture`] trait. This trait defines the logic
//! used to sample / test / benchmark the implementors of any of the `*Engine` traits defined in
//! `concrete-core`. This logic is always roughly the same and, depending on the use of the fixture,
//! boils down to:
//! ```text
//! | Iterate over a set of parameters:
//! | | Repeat multiple times:
//! | | | Generate repetition-level input prototypes.
//! | | | Sample multiple executions:
//! | | | | Generate sample-level input prototypes.
//! | | | | Synthesize actual input entities expected by the engine
//! | | | | Execute the engine
//! | | | | Compute raw outcome in a form that can be tested
//! | | | | Collect and dispose of the entities
//! | | | Compute verification criteria
//! | | | Verify that the repetition sample outcomes match the criteria
//! ```
//!
//! Note that this structure allows the generated data to be used for multiple executions or not.
//! Prototypes generated at the repetition level will be used for every samples, while prototyped
//! generated at the sample level will only be used once.
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
    /// A type containing the parameters needed to generate the execution context.
    type Parameters;

    /// A type containing the input prototypes generated at the level of the repetition (reused).
    type RepetitionPrototypes;

    /// A type containing the input prototypes generated at the level of the sample (not reused).
    type SamplePrototypes;

    /// A type containing all the objects which must exist for the engine to be executed.
    type PreExecutionContext;

    /// A type containing all the objects existing after the engine got executed.
    type PostExecutionContext;

    /// A type containing the criteria needed to perform the verification of a repetition.
    type Criteria;

    /// A type containing the outcome of an execution, such as it can be analyzed for correctness.
    type Outcome;

    /// A method which outputs an iterator over parameters.
    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>>;

    /// Generate a random set of repetition-level prototypes.
    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes;

    /// Generate a random set of sample-level prototypes.
    fn generate_random_sample_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes;

    /// A method which prepares the pre-execution context for the engine execution.
    ///
    /// The first returned value, can be used to store prototypical secret keys used in the
    /// `process_context` method to decrypt values in the output context.
    fn prepare_context(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext;

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
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome;

    /// A method which computes the verification criteria for a repetition.
    fn compute_criteria(
        parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria;

    /// A method which verify that the outcomes verify some criteria.
    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool;

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
        for _ in 0..repetitions.0 {
            let repetition_prototypes =
                Self::generate_random_repetition_prototypes(parameters, maker);
            let output = Self::test(
                maker,
                engine,
                parameters,
                &repetition_prototypes,
                sample_size,
            );
            if !output {
                return false;
            }
        }
        true
    }

    /// A method which verifies the statistical properties of a sample of engine execution, for a
    /// fixed set of raw inputs and a fixed set of parameters.
    fn test(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: &Self::Parameters,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_size: SampleSize,
    ) -> bool {
        let outputs = Self::sample(maker, engine, parameters, repetition_proto, sample_size);
        let criteria = Self::compute_criteria(parameters, maker, repetition_proto);
        Self::verify(&criteria, outputs.as_slice())
    }

    /// A method which generates a sample of engine execution, for a fixed set of raw inputs and a
    /// fixed set of parameters.
    fn sample(
        maker: &mut Maker,
        engine: &mut Engine,
        parameters: &Self::Parameters,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_size: SampleSize,
    ) -> Vec<Self::Outcome> {
        let mut outputs = Vec::with_capacity(sample_size.0);
        for _ in 0..sample_size.0 {
            let sample_proto =
                Self::generate_random_sample_prototypes(parameters, maker, repetition_proto);
            let pre_execution_context =
                Self::prepare_context(parameters, maker, repetition_proto, &sample_proto);
            let post_execution_context =
                Self::execute_engine(parameters, engine, pre_execution_context);
            let output = Self::process_context(
                parameters,
                maker,
                repetition_proto,
                &sample_proto,
                post_execution_context,
            );
            outputs.push(output);
        }
        outputs
    }
}

mod cleartext_creation;
pub use cleartext_creation::*;

mod glwe_ciphertext_discarding_encryption;
pub use glwe_ciphertext_discarding_encryption::*;

mod cleartext_retrieval;
pub use cleartext_retrieval::*;

mod cleartext_discarding_retrieval;
pub use cleartext_discarding_retrieval::*;

mod cleartext_vector_creation;
pub use cleartext_vector_creation::*;

mod cleartext_vector_discarding_retrieval;
pub use cleartext_vector_discarding_retrieval::*;

mod cleartext_vector_retrieval;
pub use cleartext_vector_retrieval::*;

mod glwe_ciphertext_trivial_decryption;
pub use glwe_ciphertext_trivial_decryption::*;

mod glwe_ciphertext_trivial_encryption;
pub use glwe_ciphertext_trivial_encryption::*;

mod glwe_ciphertext_encryption;
pub use glwe_ciphertext_encryption::*;

mod glwe_ciphertext_zero_encryption;
pub use glwe_ciphertext_zero_encryption::*;

mod glwe_ciphertext_decryption;
pub use glwe_ciphertext_decryption::*;

mod glwe_ciphertext_discarding_decryption;
pub use glwe_ciphertext_discarding_decryption::*;

mod glwe_ciphertext_vector_encryption;
pub use glwe_ciphertext_vector_encryption::*;

mod glwe_ciphertext_vector_decryption;
pub use glwe_ciphertext_vector_decryption::*;

mod glwe_ciphertext_vector_discarding_decryption;
pub use glwe_ciphertext_vector_discarding_decryption::*;

mod glwe_ciphertext_vector_discarding_encryption;
pub use glwe_ciphertext_vector_discarding_encryption::*;

mod glwe_ciphertext_vector_zero_encryption;
pub use glwe_ciphertext_vector_zero_encryption::*;

mod glwe_ciphertext_vector_trivial_decryption;
pub use glwe_ciphertext_vector_trivial_decryption::*;

mod glwe_ciphertext_vector_trivial_encryption;
pub use glwe_ciphertext_vector_trivial_encryption::*;

mod lwe_ciphertext_vector_zero_encryption;
pub use lwe_ciphertext_vector_zero_encryption::*;

mod lwe_ciphertext_encryption;
pub use lwe_ciphertext_encryption::*;

mod lwe_ciphertext_zero_encryption;
pub use lwe_ciphertext_zero_encryption::*;

mod lwe_ciphertext_decryption;
pub use lwe_ciphertext_decryption::*;

mod lwe_ciphertext_discarding_encryption;
pub use lwe_ciphertext_discarding_encryption::*;

mod lwe_ciphertext_vector_decryption;
pub use lwe_ciphertext_vector_decryption::*;

mod lwe_ciphertext_cleartext_discarding_multiplication;
pub use lwe_ciphertext_cleartext_discarding_multiplication::*;

mod lwe_ciphertext_cleartext_fusing_multiplication;
pub use lwe_ciphertext_cleartext_fusing_multiplication::*;

mod lwe_ciphertext_vector_discarding_affine_transformation;
pub use lwe_ciphertext_vector_discarding_affine_transformation::*;

mod lwe_ciphertext_vector_trivial_decryption;
pub use lwe_ciphertext_vector_trivial_decryption::*;

mod lwe_ciphertext_vector_trivial_encryption;
pub use lwe_ciphertext_vector_trivial_encryption::*;

mod lwe_ciphertext_vector_discarding_decryption;
pub use lwe_ciphertext_vector_discarding_decryption::*;

mod lwe_ciphertext_vector_discarding_encryption;
pub use lwe_ciphertext_vector_discarding_encryption::*;

mod lwe_ciphertext_discarding_keyswitch;
pub use lwe_ciphertext_discarding_keyswitch::*;

mod lwe_ciphertext_discarding_addition;
pub use lwe_ciphertext_discarding_addition::*;

mod lwe_ciphertext_discarding_negation;
pub use lwe_ciphertext_discarding_negation::*;

mod lwe_ciphertext_fusing_addition;
pub use lwe_ciphertext_fusing_addition::*;

mod lwe_ciphertext_fusing_negation;
pub use lwe_ciphertext_fusing_negation::*;

mod lwe_ciphertext_discarding_subtraction;
pub use lwe_ciphertext_discarding_subtraction::*;

mod lwe_ciphertext_fusing_subtraction;
pub use lwe_ciphertext_fusing_subtraction::*;

mod lwe_ciphertext_discarding_decryption;
pub use lwe_ciphertext_discarding_decryption::*;

mod lwe_ciphertext_plaintext_discarding_addition;
pub use lwe_ciphertext_plaintext_discarding_addition::*;

mod lwe_ciphertext_plaintext_fusing_addition;
pub use lwe_ciphertext_plaintext_fusing_addition::*;

mod lwe_ciphertext_plaintext_fusing_subtraction;
pub use lwe_ciphertext_plaintext_fusing_subtraction::*;

mod lwe_ciphertext_plaintext_discarding_subtraction;
pub use lwe_ciphertext_plaintext_discarding_subtraction::*;

mod lwe_ciphertext_vector_discarding_subtraction;
pub use lwe_ciphertext_vector_discarding_subtraction::*;

mod lwe_ciphertext_vector_encryption;
pub use lwe_ciphertext_vector_encryption::*;

mod lwe_ciphertext_vector_fusing_addition;
pub use lwe_ciphertext_vector_fusing_addition::*;

mod lwe_ciphertext_vector_discarding_addition;
pub use lwe_ciphertext_vector_discarding_addition::*;

mod lwe_ciphertext_vector_fusing_subtraction;
pub use lwe_ciphertext_vector_fusing_subtraction::*;

mod lwe_ciphertext_trivial_encryption;
pub use lwe_ciphertext_trivial_encryption::*;

mod lwe_ciphertext_trivial_decryption;
pub use lwe_ciphertext_trivial_decryption::*;

mod lwe_ciphertext_discarding_bootstrap_1;
pub use lwe_ciphertext_discarding_bootstrap_1::*;

mod lwe_ciphertext_discarding_bootstrap_2;
pub use lwe_ciphertext_discarding_bootstrap_2::*;

mod lwe_ciphertext_discarding_extraction;
pub use lwe_ciphertext_discarding_extraction::*;

mod plaintext_creation;
pub use plaintext_creation::*;

mod glwe_ciphertext_ggsw_ciphertext_discarding_external_product;
pub use glwe_ciphertext_ggsw_ciphertext_discarding_external_product::*;

mod glwe_ciphertext_ggsw_ciphertext_external_product;
pub use glwe_ciphertext_ggsw_ciphertext_external_product::*;

mod plaintext_discarding_retrieval;
pub use plaintext_discarding_retrieval::*;

mod plaintext_retrieval;
pub use plaintext_retrieval::*;

mod plaintext_vector_discarding_retrieval;
pub use plaintext_vector_discarding_retrieval::*;

mod plaintext_vector_creation;
pub use plaintext_vector_creation::*;

mod plaintext_vector_retrieval;
pub use plaintext_vector_retrieval::*;

mod lwe_keyswitch_key_creation;
pub use lwe_keyswitch_key_creation::*;

mod lwe_secret_key_creation;
pub use lwe_secret_key_creation::*;

mod glwe_secret_key_creation;
pub use glwe_secret_key_creation::*;

mod glwe_secret_key_to_lwe_secret_key_transmutation;
pub use glwe_secret_key_to_lwe_secret_key_transmutation::*;

mod lwe_bootstrap_key_creation;
pub use lwe_bootstrap_key_creation::*;

mod lwe_bootstrap_key_conversion;
pub use lwe_bootstrap_key_conversion::*;

mod lwe_bootstrap_key_discarding_conversion;
pub use lwe_bootstrap_key_discarding_conversion::*;
