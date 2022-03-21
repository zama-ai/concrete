use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{GlweCiphertext32, GlweCiphertext64};

/// A trait implemented by glwe ciphertext prototypes.
pub trait GlweCiphertextPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary glwe ciphertext entity.
pub struct ProtoBinaryGlweCiphertext32(pub(crate) GlweCiphertext32);
impl GlweCiphertextPrototype for ProtoBinaryGlweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary glwe ciphertext entity.
pub struct ProtoBinaryGlweCiphertext64(pub(crate) GlweCiphertext64);
impl GlweCiphertextPrototype for ProtoBinaryGlweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
