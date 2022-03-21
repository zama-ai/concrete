use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{LweKeyswitchKey32, LweKeyswitchKey64};

/// A trait implemented by lwe keyswitch key prototypes.
pub trait LweKeyswitchKeyPrototype {
    type InputKeyDistribution: KeyDistributionMarker;
    type OutputKeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary to binary lwe keyswitch key entity.
pub struct ProtoBinaryBinaryLweKeyswitchKey32(pub(crate) LweKeyswitchKey32);
impl LweKeyswitchKeyPrototype for ProtoBinaryBinaryLweKeyswitchKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary to binary lwe keyswitch key entity.
pub struct ProtoBinaryBinaryLweKeyswitchKey64(pub(crate) LweKeyswitchKey64);
impl LweKeyswitchKeyPrototype for ProtoBinaryBinaryLweKeyswitchKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
