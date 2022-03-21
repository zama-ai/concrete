use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::{CleartextVector32, CleartextVector64};

/// A trait implemented by cleartext vector prototypes.
pub trait CleartextVectorPrototype {
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit cleartext vector entity.
pub struct ProtoCleartextVector32(pub(crate) CleartextVector32);
impl CleartextVectorPrototype for ProtoCleartextVector32 {
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit cleartext vector entity.
pub struct ProtoCleartextVector64(pub(crate) CleartextVector64);
impl CleartextVectorPrototype for ProtoCleartextVector64 {
    type Precision = Precision64;
}
