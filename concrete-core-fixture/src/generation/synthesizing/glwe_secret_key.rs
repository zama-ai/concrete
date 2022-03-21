use crate::generation::prototyping::PrototypesGlweSecretKey;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::GlweSecretKeyEntity;

/// A trait allowing to synthesize an actual glwe secret key entity from a prototype.
pub trait SynthesizesGlweSecretKey<Precision: IntegerPrecision, GlweSecretKey>:
    PrototypesGlweSecretKey<Precision, GlweSecretKey::KeyDistribution>
where
    GlweSecretKey: GlweSecretKeyEntity,
{
    fn synthesize_glwe_secret_key(&mut self, prototype: &Self::GlweSecretKeyProto)
        -> GlweSecretKey;
    fn unsynthesize_glwe_secret_key(&mut self, entity: &GlweSecretKey) -> Self::GlweSecretKeyProto;
    fn destroy_glwe_secret_key(&mut self, entity: GlweSecretKey);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryGlweSecretKey32, ProtoBinaryGlweSecretKey64};
    use crate::generation::synthesizing::SynthesizesGlweSecretKey;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, GlweSecretKey32, GlweSecretKey64};

    impl SynthesizesGlweSecretKey<Precision32, GlweSecretKey32> for Maker {
        fn synthesize_glwe_secret_key(
            &mut self,
            prototype: &Self::GlweSecretKeyProto,
        ) -> GlweSecretKey32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_secret_key(
            &mut self,
            entity: &GlweSecretKey32,
        ) -> Self::GlweSecretKeyProto {
            ProtoBinaryGlweSecretKey32(entity.to_owned())
        }

        fn destroy_glwe_secret_key(&mut self, entity: GlweSecretKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGlweSecretKey<Precision64, GlweSecretKey64> for Maker {
        fn synthesize_glwe_secret_key(
            &mut self,
            prototype: &Self::GlweSecretKeyProto,
        ) -> GlweSecretKey64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_secret_key(
            &mut self,
            entity: &GlweSecretKey64,
        ) -> Self::GlweSecretKeyProto {
            ProtoBinaryGlweSecretKey64(entity.to_owned())
        }

        fn destroy_glwe_secret_key(&mut self, entity: GlweSecretKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
