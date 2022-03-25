use crate::generation::prototypes::{
    PackingKeyswitchKeyPrototype, ProtoBinaryBinaryPackingKeyswitchKey32,
    ProtoBinaryBinaryPackingKeyswitchKey64,
};
use crate::generation::prototyping::lwe_secret_key::PrototypesLweSecretKey;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::PackingKeyswitchKeyCreationEngine;

use super::PrototypesGlweSecretKey;

/// A trait allowing to manipulate packing keyswitch key prototypes.
pub trait PrototypesPackingKeyswitchKey<
    Precision: IntegerPrecision,
    InputKeyDistribution: KeyDistributionMarker,
    OutputKeyDistribution: KeyDistributionMarker,
>:
    PrototypesLweSecretKey<Precision, InputKeyDistribution>
    + PrototypesGlweSecretKey<Precision, OutputKeyDistribution>
{
    type PackingKeyswitchKeyProto: PackingKeyswitchKeyPrototype<
        Precision = Precision,
        InputKeyDistribution = InputKeyDistribution,
        OutputKeyDistribution = OutputKeyDistribution,
    >;
    fn new_packing_keyswitch_key(
        &mut self,
        input_key: &<Self as PrototypesLweSecretKey<
            Precision,
            InputKeyDistribution,
        >>::LweSecretKeyProto,
        output_key: &<Self as PrototypesGlweSecretKey<
            Precision,
            OutputKeyDistribution,
        >>::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::PackingKeyswitchKeyProto;
}

impl PrototypesPackingKeyswitchKey<Precision32, BinaryKeyDistribution, BinaryKeyDistribution>
    for Maker
{
    type PackingKeyswitchKeyProto = ProtoBinaryBinaryPackingKeyswitchKey32;

    fn new_packing_keyswitch_key(
        &mut self,
        input_key: &Self::LweSecretKeyProto,
        output_key: &Self::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::PackingKeyswitchKeyProto {
        ProtoBinaryBinaryPackingKeyswitchKey32(
            self.core_engine
                .create_packing_keyswitch_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_level,
                    decomposition_base_log,
                    noise,
                )
                .unwrap(),
        )
    }
}

impl PrototypesPackingKeyswitchKey<Precision64, BinaryKeyDistribution, BinaryKeyDistribution>
    for Maker
{
    type PackingKeyswitchKeyProto = ProtoBinaryBinaryPackingKeyswitchKey64;

    fn new_packing_keyswitch_key(
        &mut self,
        input_key: &Self::LweSecretKeyProto,
        output_key: &Self::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::PackingKeyswitchKeyProto {
        ProtoBinaryBinaryPackingKeyswitchKey64(
            self.core_engine
                .create_packing_keyswitch_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_level,
                    decomposition_base_log,
                    noise,
                )
                .unwrap(),
        )
    }
}
