use crate::generation::prototypes::{
    LweSecretKeyPrototype, ProtoBinaryLweSecretKey32, ProtoBinaryLweSecretKey64,
};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::LweSecretKeyCreationEngine;

/// A trait allowing to manipulate lwe secret key prototypes.
pub trait PrototypesLweSecretKey<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>
{
    type LweSecretKeyProto: LweSecretKeyPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn new_lwe_secret_key(&mut self, lwe_dimension: LweDimension) -> Self::LweSecretKeyProto;
}

impl PrototypesLweSecretKey<Precision32, BinaryKeyDistribution> for Maker {
    type LweSecretKeyProto = ProtoBinaryLweSecretKey32;

    fn new_lwe_secret_key(&mut self, lwe_dimension: LweDimension) -> Self::LweSecretKeyProto {
        ProtoBinaryLweSecretKey32(
            self.core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap(),
        )
    }
}

impl PrototypesLweSecretKey<Precision64, BinaryKeyDistribution> for Maker {
    type LweSecretKeyProto = ProtoBinaryLweSecretKey64;

    fn new_lwe_secret_key(&mut self, lwe_dimension: LweDimension) -> Self::LweSecretKeyProto {
        ProtoBinaryLweSecretKey64(
            self.core_engine
                .create_lwe_secret_key(lwe_dimension)
                .unwrap(),
        )
    }
}
