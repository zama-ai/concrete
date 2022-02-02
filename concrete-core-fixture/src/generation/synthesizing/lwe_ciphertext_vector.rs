use crate::generation::prototyping::PrototypesLweCiphertextVector;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::LweCiphertextVectorEntity;

/// A trait allowing to synthesize an actual lwe ciphertext vector entity from a prototype.
pub trait SynthesizesLweCiphertextVector<Precision: IntegerPrecision, LweCiphertextVector>:
    PrototypesLweCiphertextVector<Precision, LweCiphertextVector::KeyDistribution>
where
    LweCiphertextVector: LweCiphertextVectorEntity,
{
    fn synthesize_lwe_ciphertext_vector(
        &mut self,
        prototype: &Self::LweCiphertextVectorProto,
    ) -> LweCiphertextVector;
    fn unsynthesize_lwe_ciphertext_vector(
        &mut self,
        entity: &LweCiphertextVector,
    ) -> Self::LweCiphertextVectorProto;
    fn destroy_lwe_ciphertext_vector(&mut self, entity: LweCiphertextVector);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{
        ProtoBinaryLweCiphertextVector32, ProtoBinaryLweCiphertextVector64,
    };
    use crate::generation::synthesizing::SynthesizesLweCiphertextVector;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, LweCiphertextVector32, LweCiphertextVector64};

    impl SynthesizesLweCiphertextVector<Precision32, LweCiphertextVector32> for Maker {
        fn synthesize_lwe_ciphertext_vector(
            &mut self,
            prototype: &Self::LweCiphertextVectorProto,
        ) -> LweCiphertextVector32 {
            prototype.0.to_owned()
        }
        fn unsynthesize_lwe_ciphertext_vector(
            &mut self,
            entity: &LweCiphertextVector32,
        ) -> Self::LweCiphertextVectorProto {
            ProtoBinaryLweCiphertextVector32(entity.to_owned())
        }
        fn destroy_lwe_ciphertext_vector(&mut self, entity: LweCiphertextVector32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesLweCiphertextVector<Precision64, LweCiphertextVector64> for Maker {
        fn synthesize_lwe_ciphertext_vector(
            &mut self,
            prototype: &Self::LweCiphertextVectorProto,
        ) -> LweCiphertextVector64 {
            prototype.0.to_owned()
        }
        fn unsynthesize_lwe_ciphertext_vector(
            &mut self,
            entity: &LweCiphertextVector64,
        ) -> Self::LweCiphertextVectorProto {
            ProtoBinaryLweCiphertextVector64(entity.to_owned())
        }
        fn destroy_lwe_ciphertext_vector(&mut self, entity: LweCiphertextVector64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
