//! A module manipulating objects in the raw message space.
//!
//! For all the tests present in this application, we need to be able to generate input plaintext,
//! and analyze output plaintexts. We implement those generation and analysis function only for the
//! _raw_ `u32` and `u64` types.
//!
//! In particular, this module allows to perform the very beginning and the very end of a test:
//!
//! + Generation of raw input messages using different distributions (see [`generation`])
//! + Statistical analysis of the properties of messages after the operation (see
//! [`statistical_test`])
//!
//! The rest of the test is deferred to the [`crate::prototyping`] module.

pub mod generation;
pub mod statistical_test;
