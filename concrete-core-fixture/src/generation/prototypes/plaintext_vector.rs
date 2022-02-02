use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::{PlaintextVector32, PlaintextVector64};

/// A trait implemented by plaintext vector prototypes.
pub trait PlaintextVectorPrototype {
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit plaintext vector entity.
pub struct ProtoPlaintextVector32(pub(crate) PlaintextVector32);
impl PlaintextVectorPrototype for ProtoPlaintextVector32 {
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit plaintext vector entity.
pub struct ProtoPlaintextVector64(pub(crate) PlaintextVector64);
impl PlaintextVectorPrototype for ProtoPlaintextVector64 {
    type Precision = Precision64;
}
