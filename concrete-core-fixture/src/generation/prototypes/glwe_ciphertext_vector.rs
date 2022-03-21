use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{GlweCiphertextVector32, GlweCiphertextVector64};

/// A trait implemented by glwe ciphertext vector prototypes.
pub trait GlweCiphertextVectorPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary glwe ciphertext vector entity.
pub struct ProtoBinaryGlweCiphertextVector32(pub(crate) GlweCiphertextVector32);
impl GlweCiphertextVectorPrototype for ProtoBinaryGlweCiphertextVector32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary glwe ciphertext vector entity.
pub struct ProtoBinaryGlweCiphertextVector64(pub(crate) GlweCiphertextVector64);
impl GlweCiphertextVectorPrototype for ProtoBinaryGlweCiphertextVector64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
