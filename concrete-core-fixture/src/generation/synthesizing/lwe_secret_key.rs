use crate::generation::prototyping::PrototypesLweSecretKey;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::LweSecretKeyEntity;

/// A trait allowing to synthesize an actual lwe secret key vector entity from a prototype.
pub trait SynthesizesLweSecretKey<Precision: IntegerPrecision, LweSecretKey>:
    PrototypesLweSecretKey<Precision, LweSecretKey::KeyDistribution>
where
    LweSecretKey: LweSecretKeyEntity,
{
    fn synthesize_lwe_secret_key(&mut self, prototype: &Self::LweSecretKeyProto) -> LweSecretKey;
    fn unsynthesize_lwe_secret_key(&mut self, entity: &LweSecretKey) -> Self::LweSecretKeyProto;
    fn destroy_lwe_secret_key(&mut self, entity: LweSecretKey);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryLweSecretKey32, ProtoBinaryLweSecretKey64};
    use crate::generation::synthesizing::SynthesizesLweSecretKey;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, LweSecretKey32, LweSecretKey64};

    impl SynthesizesLweSecretKey<Precision32, LweSecretKey32> for Maker {
        fn synthesize_lwe_secret_key(
            &mut self,
            prototype: &Self::LweSecretKeyProto,
        ) -> LweSecretKey32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_secret_key(
            &mut self,
            entity: &LweSecretKey32,
        ) -> Self::LweSecretKeyProto {
            ProtoBinaryLweSecretKey32(entity.to_owned())
        }
        fn destroy_lwe_secret_key(&mut self, entity: LweSecretKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweSecretKey<Precision64, LweSecretKey64> for Maker {
        fn synthesize_lwe_secret_key(
            &mut self,
            prototype: &Self::LweSecretKeyProto,
        ) -> LweSecretKey64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_secret_key(
            &mut self,
            entity: &LweSecretKey64,
        ) -> Self::LweSecretKeyProto {
            ProtoBinaryLweSecretKey64(entity.to_owned())
        }

        fn destroy_lwe_secret_key(&mut self, entity: LweSecretKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
