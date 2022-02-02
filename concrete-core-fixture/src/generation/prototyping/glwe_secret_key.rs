use crate::generation::prototypes::{
    GlweSecretKeyPrototype, ProtoBinaryGlweSecretKey32, ProtoBinaryGlweSecretKey64,
};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::prelude::markers::{BinaryKeyDistribution, KeyDistributionMarker};
use concrete_core::prelude::GlweSecretKeyCreationEngine;

/// A trait allowing to manipulate GLWE secret key prototypes.
pub trait PrototypesGlweSecretKey<
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
>
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
}
