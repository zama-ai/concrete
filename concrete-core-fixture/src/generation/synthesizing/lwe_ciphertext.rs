use crate::generation::prototyping::PrototypesLweCiphertext;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::LweCiphertextEntity;

/// A trait allowing to synthesize an actual lwe ciphertext entity from a prototype.
pub trait SynthesizesLweCiphertext<Precision: IntegerPrecision, LweCiphertext>:
    PrototypesLweCiphertext<Precision, LweCiphertext::KeyDistribution>
where
    LweCiphertext: LweCiphertextEntity,
{
    fn synthesize_lwe_ciphertext(&mut self, prototype: &Self::LweCiphertextProto) -> LweCiphertext;
    fn unsynthesize_lwe_ciphertext(&mut self, entity: &LweCiphertext) -> Self::LweCiphertextProto;
    fn destroy_lwe_ciphertext(&mut self, entity: LweCiphertext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryLweCiphertext32, ProtoBinaryLweCiphertext64};
    use crate::generation::synthesizing::SynthesizesLweCiphertext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, LweCiphertext32, LweCiphertext64};

    impl SynthesizesLweCiphertext<Precision32, LweCiphertext32> for Maker {
        fn synthesize_lwe_ciphertext(
            &mut self,
            prototype: &Self::LweCiphertextProto,
        ) -> LweCiphertext32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_ciphertext(
            &mut self,
            entity: &LweCiphertext32,
        ) -> Self::LweCiphertextProto {
            ProtoBinaryLweCiphertext32(entity.to_owned())
        }

        fn destroy_lwe_ciphertext(&mut self, entity: LweCiphertext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweCiphertext<Precision64, LweCiphertext64> for Maker {
        fn synthesize_lwe_ciphertext(
            &mut self,
            prototype: &Self::LweCiphertextProto,
        ) -> LweCiphertext64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_ciphertext(
            &mut self,
            entity: &LweCiphertext64,
        ) -> Self::LweCiphertextProto {
            ProtoBinaryLweCiphertext64(entity.to_owned())
        }

        fn destroy_lwe_ciphertext(&mut self, entity: LweCiphertext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
