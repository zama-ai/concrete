use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{LweCiphertextVector32, LweCiphertextVector64};

/// A trait implemented by lwe ciphertext vector prototypes.
pub trait LweCiphertextVectorPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary lwe ciphertext vector entity.
pub struct ProtoBinaryLweCiphertextVector32(pub(crate) LweCiphertextVector32);
impl LweCiphertextVectorPrototype for ProtoBinaryLweCiphertextVector32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

///  type representing the prototype of a 64 bit binary lwe ciphertext vector entity.
pub struct ProtoBinaryLweCiphertextVector64(pub(crate) LweCiphertextVector64);
impl LweCiphertextVectorPrototype for ProtoBinaryLweCiphertextVector64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
