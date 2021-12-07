//! A library containing generic statistical tests for `concrete-core` operators.
//!
//! This application contains generic testing functions, aimed at verifying that the statistical
//! properties of the operators implemented in `concrete_core` follow our models (embodied by the
//! formulas implemented in `concrete_npe`)
//!
//! The generic nature of the test functions implemented here, makes it possible to test every
//! operators of the `concrete-core` library, using the same functions. Then, testing a new backend
//! mainly consists in appropriately instantiating the tests.
//!
//! # Test architecture
//!
//! Basically all the tests go through the same steps:
//! ```ascii
//! -> Generate raw messages
//! -----> Turn raw input messages to prototypical input plaintexts
//! -----> Manipulate the prototypical input entities (encryption, pre-test operations, ..)
//! ------------> Synthesize the actual input entities from the prototypical input entities
//! ---------------------> Call the tested engine on the actual input entities
//! <------------ Retrieve the prototypical output entities from the actual output entities
//! <----- Manipulate the prototypical output entities (decryption, post-test operations, ...)
//! <----- Retrieve the raw output messages from the prototypical output plaintexts
//! <- Statistically test on the raw output message
//! ```
//!
//! # Reading the code
//!
//! The [`raw`] module handles:
//! ```ascii
//! -> Generate raw messages
//! <- Statistically test on the raw output message
//! ```
//!
//! The [`prototyping`] module handles:
//! ```ascii
//! -----> Turn raw input messages to prototypical input plaintexts
//! -----> Manipulate the prototypical input entities (encryption, pre-test operations, ..)
//! <----- Manipulate the prototypical output entities (decryption, post-test operations, ...)
//! <----- Retrieve the raw output messages from the prototypical output plaintexts
//! ```
//!
//! The [`synthesizing`] module handles:
//! ```ascii
//! ------------> Synthesize the actual input entities from the prototypical input entities
//! <------------ Retrieve the prototypical output entities from the actual output entities
//! ```
//!
//! The [`generics`] module contains function which articulate all the aforementioned steps plus:
//! ```ascii
//! ---------------------> Call the tested engine on the actual input entities
//! ```

use concrete_core::prelude::AbstractEngine;

pub mod backends;
pub mod generics;
pub mod prototyping;
pub mod raw;
pub mod synthesizing;

/// A type representing the number of times we repeat a test for a given set of parameters.
pub struct Repetitions(pub usize);

/// A type representing the number of samples needed to perform a statistical test.
pub struct SampleSize(pub usize);

/// The number of time a test is repeated for a single set of parameter.
pub const REPETITIONS: Repetitions = Repetitions(10);

/// The size of the sample used to perform statistical tests.
pub const SAMPLE_SIZE: SampleSize = SampleSize(1000);

/// The central structure used to generate the input data for all the tests.
///
/// This structure contains the necessary tools to:
///
/// + Convert back and forth between raw integer types and prototypical plaintexts.
/// + Manipulate prototypical entities, to generate compatible prototypical inputs for tests.
/// + Convert back and forth between prototypical entities and actual entity type used for the
/// tests.
pub struct Maker {
    #[cfg(feature = "backend_core")]
    core_engine: concrete_core::backends::core::engines::CoreEngine,
}

impl Default for Maker {
    fn default() -> Self {
        Maker {
            #[cfg(feature = "backend_core")]
            core_engine: concrete_core::backends::core::engines::CoreEngine::new().unwrap(),
        }
    }
}
