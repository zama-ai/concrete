use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{LweBootstrapKey32, LweBootstrapKey64};

/// A trait implemented by lwe bootstrap key prototypes.
pub trait LweBootstrapKeyPrototype {
    type InputKeyDistribution: KeyDistributionMarker;
    type OutputKeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary to binary lwe bootstrap key entity.
pub struct ProtoBinaryBinaryLweBootstrapKey32(pub(crate) LweBootstrapKey32);
impl LweBootstrapKeyPrototype for ProtoBinaryBinaryLweBootstrapKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary to binary lwe bootstrap key entity.
pub struct ProtoBinaryBinaryLweBootstrapKey64(pub(crate) LweBootstrapKey64);
impl LweBootstrapKeyPrototype for ProtoBinaryBinaryLweBootstrapKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
