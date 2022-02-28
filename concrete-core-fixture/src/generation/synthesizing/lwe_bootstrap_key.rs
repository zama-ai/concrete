use crate::generation::prototyping::PrototypesLweBootstrapKey;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::LweBootstrapKeyEntity;

/// A trait allowing to synthesize an actual lwe bootstrap key entity from a prototype.
pub trait SynthesizesLweBootstrapKey<Precision: IntegerPrecision, LweBootstrapKey>:
    PrototypesLweBootstrapKey<
    Precision,
    LweBootstrapKey::InputKeyDistribution,
    LweBootstrapKey::OutputKeyDistribution,
>
where
    LweBootstrapKey: LweBootstrapKeyEntity,
{
    fn synthesize_lwe_bootstrap_key(
        &mut self,
        prototype: &Self::LweBootstrapKeyProto,
    ) -> LweBootstrapKey;
    fn unsynthesize_lwe_bootstrap_key(
        &mut self,
        entity: &LweBootstrapKey,
    ) -> Self::LweBootstrapKeyProto;
    fn destroy_lwe_bootstrap_key(&mut self, entity: LweBootstrapKey);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{
        ProtoBinaryBinaryLweBootstrapKey32, ProtoBinaryBinaryLweBootstrapKey64,
    };
    use crate::generation::synthesizing::SynthesizesLweBootstrapKey;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{
        DestructionEngine, FourierLweBootstrapKey32, FourierLweBootstrapKey64, LweBootstrapKey32,
        LweBootstrapKey64, LweBootstrapKeyConversionEngine,
    };

    impl SynthesizesLweBootstrapKey<Precision32, LweBootstrapKey32> for Maker {
        fn synthesize_lwe_bootstrap_key(
            &mut self,
            prototype: &Self::LweBootstrapKeyProto,
        ) -> LweBootstrapKey32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_bootstrap_key(
            &mut self,
            entity: &LweBootstrapKey32,
        ) -> Self::LweBootstrapKeyProto {
            ProtoBinaryBinaryLweBootstrapKey32(entity.to_owned())
        }

        fn destroy_lwe_bootstrap_key(&mut self, entity: LweBootstrapKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweBootstrapKey<Precision64, LweBootstrapKey64> for Maker {
        fn synthesize_lwe_bootstrap_key(
            &mut self,
            prototype: &Self::LweBootstrapKeyProto,
        ) -> LweBootstrapKey64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_bootstrap_key(
            &mut self,
            entity: &LweBootstrapKey64,
        ) -> Self::LweBootstrapKeyProto {
            ProtoBinaryBinaryLweBootstrapKey64(entity.to_owned())
        }

        fn destroy_lwe_bootstrap_key(&mut self, entity: LweBootstrapKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweBootstrapKey<Precision32, FourierLweBootstrapKey32> for Maker {
        fn synthesize_lwe_bootstrap_key(
            &mut self,
            prototype: &Self::LweBootstrapKeyProto,
        ) -> FourierLweBootstrapKey32 {
            self.core_engine
                .convert_lwe_bootstrap_key(&prototype.0)
                .unwrap()
        }

        fn unsynthesize_lwe_bootstrap_key(
            &mut self,
            _entity: &FourierLweBootstrapKey32,
        ) -> Self::LweBootstrapKeyProto {
            todo!()
        }

        fn destroy_lwe_bootstrap_key(&mut self, entity: FourierLweBootstrapKey32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweBootstrapKey<Precision64, FourierLweBootstrapKey64> for Maker {
        fn synthesize_lwe_bootstrap_key(
            &mut self,
            prototype: &Self::LweBootstrapKeyProto,
        ) -> FourierLweBootstrapKey64 {
            self.core_engine
                .convert_lwe_bootstrap_key(&prototype.0)
                .unwrap()
        }

        fn unsynthesize_lwe_bootstrap_key(
            &mut self,
            _entity: &FourierLweBootstrapKey64,
        ) -> Self::LweBootstrapKeyProto {
            todo!()
        }

        fn destroy_lwe_bootstrap_key(&mut self, entity: FourierLweBootstrapKey64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
