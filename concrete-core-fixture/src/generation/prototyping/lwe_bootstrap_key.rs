use crate::generation::prototypes::{
    LweBootstrapKeyPrototype, ProtoBinaryBinaryLweBootstrapKey32,
    ProtoBinaryBinaryLweBootstrapKey64,
};
use crate::generation::prototyping::glwe_secret_key::PrototypesGlweSecretKey;
use crate::generation::prototyping::lwe_secret_key::PrototypesLweSecretKey;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::LweBootstrapKeyCreationEngine;

/// A trait allowing to manipulate LWE bootstrap key prototypes.
pub trait PrototypesLweBootstrapKey<
    Precision: IntegerPrecision,
    InputKeyDistribution: KeyDistributionMarker,
    OutputKeyDistribution: KeyDistributionMarker,
>:
    PrototypesLweSecretKey<Precision, InputKeyDistribution>
    + PrototypesGlweSecretKey<Precision, OutputKeyDistribution>
{
    type LweBootstrapKeyProto: LweBootstrapKeyPrototype<
        Precision = Precision,
        InputKeyDistribution = InputKeyDistribution,
        OutputKeyDistribution = OutputKeyDistribution,
    >;
    fn new_lwe_bootstrap_key(
        &mut self,
        input_key: &<Self as PrototypesLweSecretKey<Precision, InputKeyDistribution>>::LweSecretKeyProto,
        output_key: &Self::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::LweBootstrapKeyProto;
}

impl PrototypesLweBootstrapKey<Precision32, BinaryKeyDistribution, BinaryKeyDistribution>
    for Maker
{
    type LweBootstrapKeyProto = ProtoBinaryBinaryLweBootstrapKey32;

    fn new_lwe_bootstrap_key(
        &mut self,
        input_key: &Self::LweSecretKeyProto,
        output_key: &Self::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::LweBootstrapKeyProto {
        ProtoBinaryBinaryLweBootstrapKey32(
            self.core_engine
                .create_lwe_bootstrap_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_base_log,
                    decomposition_level,
                    noise,
                )
                .unwrap(),
        )
    }
}

impl PrototypesLweBootstrapKey<Precision64, BinaryKeyDistribution, BinaryKeyDistribution>
    for Maker
{
    type LweBootstrapKeyProto = ProtoBinaryBinaryLweBootstrapKey64;

    fn new_lwe_bootstrap_key(
        &mut self,
        input_key: &Self::LweSecretKeyProto,
        output_key: &Self::GlweSecretKeyProto,
        decomposition_level: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Self::LweBootstrapKeyProto {
        ProtoBinaryBinaryLweBootstrapKey64(
            self.core_engine
                .create_lwe_bootstrap_key(
                    &input_key.0,
                    &output_key.0,
                    decomposition_base_log,
                    decomposition_level,
                    noise,
                )
                .unwrap(),
        )
    }
}
