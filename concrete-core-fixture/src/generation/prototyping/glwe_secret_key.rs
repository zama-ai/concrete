use crate::generation::prototypes::{
    GlweSecretKeyPrototype, ProtoBinaryGlweSecretKey32, ProtoBinaryGlweSecretKey64,
    ProtoBinaryLweSecretKey32, ProtoBinaryLweSecretKey64,
};
use crate::generation::prototyping::PrototypesLweSecretKey;
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::{GlweSecretKeyCreationEngine, GlweToLweSecretKeyTransmutationEngine};

/// A trait allowing to manipulate GLWE secret key prototypes.
pub trait PrototypesGlweSecretKey<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>: PrototypesLweSecretKey<Precision, KeyDistribution>
{
    type GlweSecretKeyProto: GlweSecretKeyPrototype<
        Precision = Precision,
        KeyDistribution = KeyDistribution,
    >;
    fn new_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::GlweSecretKeyProto;
    fn convert_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        input: &Self::GlweSecretKeyProto,
    ) -> Self::LweSecretKeyProto;
}

impl PrototypesGlweSecretKey<Precision32, BinaryKeyDistribution> for Maker {
    type GlweSecretKeyProto = ProtoBinaryGlweSecretKey32;

    fn new_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::GlweSecretKeyProto {
        ProtoBinaryGlweSecretKey32(
            self.core_engine
                .create_glwe_secret_key(glwe_dimension, polynomial_size)
                .unwrap(),
        )
    }

    fn convert_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        input: &Self::GlweSecretKeyProto,
    ) -> Self::LweSecretKeyProto {
        ProtoBinaryLweSecretKey32(
            self.core_engine
                .transmute_glwe_secret_key_to_lwe_secret_key(input.0.to_owned())
                .unwrap(),
        )
    }
}

impl PrototypesGlweSecretKey<Precision64, BinaryKeyDistribution> for Maker {
    type GlweSecretKeyProto = ProtoBinaryGlweSecretKey64;

    fn new_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Self::GlweSecretKeyProto {
        ProtoBinaryGlweSecretKey64(
            self.core_engine
                .create_glwe_secret_key(glwe_dimension, polynomial_size)
                .unwrap(),
        )
    }

    fn convert_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        input: &Self::GlweSecretKeyProto,
    ) -> Self::LweSecretKeyProto {
        ProtoBinaryLweSecretKey64(
            self.core_engine
                .transmute_glwe_secret_key_to_lwe_secret_key(input.0.to_owned())
                .unwrap(),
        )
    }
}
