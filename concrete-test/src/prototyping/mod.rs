//! A module manipulating objects in the prototype space.
//!
//! Generating the entities used as inputs to engine methods usually requires several steps to
//! ensure the compatibility on inputs (same secret keys for instance). Maintaining a separate
//! function to generate every combination of entity types implemented by `concrete_core` backends
//! is not possible. For this reason, all those manipulations are performed on _prototypical_
//! entities, before being transformed to actual entities by the [`crate::synthesizing`] module.
//!
//! In particular, this module allows to address the following steps of a test:
//!
//! + Conversion between raw messages and prototypical plaintexts
//! + Manipulation of prototypical entities (encryption/decryption, key generation, pre/post-test
//! operations, ..)
//!
//! The rest of the test is deferred to the [`crate::synthesizing`] module.
use crate::raw::generation::RawUnsignedIntegers;

pub mod prototyper;
pub mod prototypes;

/// A trait for marker type representing integer precision managed in `concrete_core`.
pub trait IntegerPrecision {
    type Raw: RawUnsignedIntegers;
}

/// A type representing the 32 bits precision for integers.
pub struct Precision32;
impl IntegerPrecision for Precision32 {
    type Raw = u32;
}

/// A type representing the 64 bits precision for integers.
pub struct Precision64;
impl IntegerPrecision for Precision64 {
    type Raw = u64;
}
