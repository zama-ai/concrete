use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{GlweSecretKey32, GlweSecretKey64};

/// A trait implemented by glwe secret key prototypes.
pub trait GlweSecretKeyPrototype {
    type KeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary glwe secret key entity.
pub struct ProtoBinaryGlweSecretKey32(pub(crate) GlweSecretKey32);
impl GlweSecretKeyPrototype for ProtoBinaryGlweSecretKey32 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary glwe secret key entity.
pub struct ProtoBinaryGlweSecretKey64(pub(crate) GlweSecretKey64);
impl GlweSecretKeyPrototype for ProtoBinaryGlweSecretKey64 {
    type KeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
