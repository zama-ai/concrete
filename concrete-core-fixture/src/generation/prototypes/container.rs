use crate::generation::{IntegerPrecision, Precision32, Precision64};

/// A trait implemented by container prototypes.
pub trait ContainerPrototype {
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a Vec containing unsigned 32 bit integers.
pub struct ProtoVec32(pub(crate) Vec<u32>);
impl ContainerPrototype for ProtoVec32 {
    type Precision = Precision32;
}

/// A type representing the prototype of a Vec containing unsigned 64 bit integers.
pub struct ProtoVec64(pub(crate) Vec<u64>);
impl ContainerPrototype for ProtoVec64 {
    type Precision = Precision64;
}
