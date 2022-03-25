use crate::generation::prototyping::PrototypesPackingKeyswitchKey;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::PackingKeyswitchKeyEntity;

pub trait SynthesizesPackingKeyswitchKey<Precision: IntegerPrecision, PackingKeyswitchKey>:
    PrototypesPackingKeyswitchKey<
    Precision,
    PackingKeyswitchKey::InputKeyDistribution,
    PackingKeyswitchKey::OutputKeyDistribution,
>
where
    PackingKeyswitchKey: PackingKeyswitchKeyEntity,
{
    fn synthesize_packing_keyswitch_key(
        &mut self,
        prototype: &Self::PackingKeyswitchKeyProto,
    ) -> PackingKeyswitchKey;
    fn unsynthesize_packing_keyswitch_key(
        &mut self,
        entity: &PackingKeyswitchKey,
    ) -> Self::PackingKeyswitchKeyProto;
    fn destroy_packing_keyswitch_key(&mut self, entity: PackingKeyswitchKey);
}

// #[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{
        ProtoBinaryBinaryPackingKeyswitchKey32, ProtoBinaryBinaryPackingKeyswitchKey64,
    };
    use crate::generation::synthesizing::SynthesizesPackingKeyswitchKey;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, PackingKeyswitchKey32, PackingKeyswitchKey64};

    impl SynthesizesPackingKeyswitchKey<Precision32, PackingKeyswitchKey32> for Maker {
        fn synthesize_packing_keyswitch_key(
            &mut self,
            prototype: &Self::PackingKeyswitchKeyProto,
        ) -> PackingKeyswitchKey32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_packing_keyswitch_key(
            &mut self,
            entity: &PackingKeyswitchKey32,
        ) -> Self::PackingKeyswitchKeyProto {
            ProtoBinaryBinaryPackingKeyswitchKey32(entity.to_owned())
        }

        fn destroy_packing_keyswitch_key(&mut self, entity: PackingKeyswitchKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesPackingKeyswitchKey<Precision64, PackingKeyswitchKey64> for Maker {
        fn synthesize_packing_keyswitch_key(
            &mut self,
            prototype: &Self::PackingKeyswitchKeyProto,
        ) -> PackingKeyswitchKey64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_packing_keyswitch_key(
            &mut self,
            entity: &PackingKeyswitchKey64,
        ) -> Self::PackingKeyswitchKeyProto {
            ProtoBinaryBinaryPackingKeyswitchKey64(entity.to_owned())
        }

        fn destroy_packing_keyswitch_key(&mut self, entity: PackingKeyswitchKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
