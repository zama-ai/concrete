use crate::generation::prototyping::PrototypesLweKeyswitchKey;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::LweKeyswitchKeyEntity;

pub trait SynthesizesLweKeyswitchKey<Precision: IntegerPrecision, LweKeyswitchKey>:
    PrototypesLweKeyswitchKey<
    Precision,
    LweKeyswitchKey::InputKeyDistribution,
    LweKeyswitchKey::OutputKeyDistribution,
>
where
    LweKeyswitchKey: LweKeyswitchKeyEntity,
{
    fn synthesize_lwe_keyswitch_key(
        &mut self,
        prototype: &Self::LweKeyswitchKeyProto,
    ) -> LweKeyswitchKey;
    fn unsynthesize_lwe_keyswitch_key(
        &mut self,
        entity: &LweKeyswitchKey,
    ) -> Self::LweKeyswitchKeyProto;
    fn destroy_lwe_keyswitch_key(&mut self, entity: LweKeyswitchKey);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{
        ProtoBinaryBinaryLweKeyswitchKey32, ProtoBinaryBinaryLweKeyswitchKey64,
    };
    use crate::generation::synthesizing::SynthesizesLweKeyswitchKey;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, LweKeyswitchKey32, LweKeyswitchKey64};

    impl SynthesizesLweKeyswitchKey<Precision32, LweKeyswitchKey32> for Maker {
        fn synthesize_lwe_keyswitch_key(
            &mut self,
            prototype: &Self::LweKeyswitchKeyProto,
        ) -> LweKeyswitchKey32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_keyswitch_key(
            &mut self,
            entity: &LweKeyswitchKey32,
        ) -> Self::LweKeyswitchKeyProto {
            ProtoBinaryBinaryLweKeyswitchKey32(entity.to_owned())
        }

        fn destroy_lwe_keyswitch_key(&mut self, entity: LweKeyswitchKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweKeyswitchKey<Precision64, LweKeyswitchKey64> for Maker {
        fn synthesize_lwe_keyswitch_key(
            &mut self,
            prototype: &Self::LweKeyswitchKeyProto,
        ) -> LweKeyswitchKey64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_keyswitch_key(
            &mut self,
            entity: &LweKeyswitchKey64,
        ) -> Self::LweKeyswitchKeyProto {
            ProtoBinaryBinaryLweKeyswitchKey64(entity.to_owned())
        }

        fn destroy_lwe_keyswitch_key(&mut self, entity: LweKeyswitchKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
