use crate::generation::{IntegerPrecision, Precision32, Precision64};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{PackingKeyswitchKey32, PackingKeyswitchKey64};

/// A trait implemented by packing keyswitch key prototypes.
pub trait PackingKeyswitchKeyPrototype {
    type InputKeyDistribution: KeyDistributionMarker;
    type OutputKeyDistribution: KeyDistributionMarker;
    type Precision: IntegerPrecision;
}

/// A type representing the prototype of a 32 bit binary to binary packing keyswitch key entity.
pub struct ProtoBinaryBinaryPackingKeyswitchKey32(pub(crate) PackingKeyswitchKey32);
impl PackingKeyswitchKeyPrototype for ProtoBinaryBinaryPackingKeyswitchKey32 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision32;
}

/// A type representing the prototype of a 64 bit binary to binary packing keyswitch key entity.
pub struct ProtoBinaryBinaryPackingKeyswitchKey64(pub(crate) PackingKeyswitchKey64);
impl PackingKeyswitchKeyPrototype for ProtoBinaryBinaryPackingKeyswitchKey64 {
    type InputKeyDistribution = BinaryKeyDistribution;
    type OutputKeyDistribution = BinaryKeyDistribution;
    type Precision = Precision64;
}
