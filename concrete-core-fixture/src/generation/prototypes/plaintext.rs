use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::{Plaintext32, Plaintext64};

/// A trait implemented by plaintext prototypes.
pub trait PlaintextPrototype {
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit plaintext entity.
pub struct ProtoPlaintext32(pub(crate) Plaintext32);
impl PlaintextPrototype for ProtoPlaintext32 {
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit plaintext entity.
pub struct ProtoPlaintext64(pub(crate) Plaintext64);
impl PlaintextPrototype for ProtoPlaintext64 {
    type Precision = Precision64;
}
