use crate::generation::prototyping::PrototypesGlweCiphertextVector;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::GlweCiphertextVectorEntity;

/// A trait allowing to synthesize an actual glwe ciphertext vector entity from a prototype.
pub trait SynthesizesGlweCiphertextVector<Precision: IntegerPrecision, GlweCiphertextVector>:
    PrototypesGlweCiphertextVector<Precision, GlweCiphertextVector::KeyDistribution>
where
    GlweCiphertextVector: GlweCiphertextVectorEntity,
{
    fn synthesize_glwe_ciphertext_vector(
        &mut self,
        prototype: &Self::GlweCiphertextVectorProto,
    ) -> GlweCiphertextVector;
    fn unsynthesize_glwe_ciphertext_vector(
        &mut self,
        entity: &GlweCiphertextVector,
    ) -> Self::GlweCiphertextVectorProto;
    fn destroy_glwe_ciphertext_vector(&mut self, entity: GlweCiphertextVector);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{
        ProtoBinaryGlweCiphertextVector32, ProtoBinaryGlweCiphertextVector64,
    };
    use crate::generation::synthesizing::SynthesizesGlweCiphertextVector;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{
        DestructionEngine, GlweCiphertextVector32, GlweCiphertextVector64,
    };

    impl SynthesizesGlweCiphertextVector<Precision32, GlweCiphertextVector32> for Maker {
        fn synthesize_glwe_ciphertext_vector(
            &mut self,
            prototype: &Self::GlweCiphertextVectorProto,
        ) -> GlweCiphertextVector32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_ciphertext_vector(
            &mut self,
            entity: &GlweCiphertextVector32,
        ) -> Self::GlweCiphertextVectorProto {
            ProtoBinaryGlweCiphertextVector32(entity.to_owned())
        }

        fn destroy_glwe_ciphertext_vector(&mut self, entity: GlweCiphertextVector32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGlweCiphertextVector<Precision64, GlweCiphertextVector64> for Maker {
        fn synthesize_glwe_ciphertext_vector(
            &mut self,
            prototype: &Self::GlweCiphertextVectorProto,
        ) -> GlweCiphertextVector64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_ciphertext_vector(
            &mut self,
            entity: &GlweCiphertextVector64,
        ) -> Self::GlweCiphertextVectorProto {
            ProtoBinaryGlweCiphertextVector64(entity.to_owned())
        }

        fn destroy_glwe_ciphertext_vector(&mut self, entity: GlweCiphertextVector64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
