use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{GgswCiphertext32, GgswCiphertext64};

/// A trait implemented by ggsw ciphertext prototypes.
pub trait GgswCiphertextPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary ggsw ciphertext entity.
pub struct ProtoBinaryGgswCiphertext32(pub(crate) GgswCiphertext32);
impl GgswCiphertextPrototype for ProtoBinaryGgswCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary glwe ciphertext entity.
pub struct ProtoBinaryGgswCiphertext64(pub(crate) GgswCiphertext64);
impl GgswCiphertextPrototype for ProtoBinaryGgswCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
