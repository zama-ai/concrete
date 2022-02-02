use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{LweCiphertext32, LweCiphertext64};

/// A trait implemented by lwe ciphertext prototypes.
pub trait LweCiphertextPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary lwe ciphertext entity.
pub struct ProtoBinaryLweCiphertext32(pub(crate) LweCiphertext32);
impl LweCiphertextPrototype for ProtoBinaryLweCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary lwe ciphertext entity.
pub struct ProtoBinaryLweCiphertext64(pub(crate) LweCiphertext64);
impl LweCiphertextPrototype for ProtoBinaryLweCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
