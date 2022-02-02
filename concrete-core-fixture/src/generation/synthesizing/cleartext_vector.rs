use crate::generation::prototyping::PrototypesCleartextVector;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::CleartextVectorEntity;

/// A trait allowing to synthesize an actual cleartext vector entity from a prototype.
pub trait SynthesizesCleartextVector<Precision: IntegerPrecision, CleartextVector>:
    PrototypesCleartextVector<Precision>
where
    CleartextVector: CleartextVectorEntity,
{
    fn synthesize_cleartext_vector(
        &mut self,
        prototype: &Self::CleartextVectorProto,
    ) -> CleartextVector;
    fn unsynthesize_cleartext_vector(
        &mut self,
        entity: &CleartextVector,
    ) -> Self::CleartextVectorProto;
    fn destroy_cleartext_vector(&mut self, entity: CleartextVector);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoCleartextVector32, ProtoCleartextVector64};
    use crate::generation::synthesizing::SynthesizesCleartextVector;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{CleartextVector32, CleartextVector64, DestructionEngine};

    impl SynthesizesCleartextVector<Precision32, CleartextVector32> for Maker {
        fn synthesize_cleartext_vector(
            &mut self,
            prototype: &Self::CleartextVectorProto,
        ) -> CleartextVector32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_cleartext_vector(
            &mut self,
            entity: &CleartextVector32,
        ) -> Self::CleartextVectorProto {
            ProtoCleartextVector32(entity.to_owned())
        }
        fn destroy_cleartext_vector(&mut self, entity: CleartextVector32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesCleartextVector<Precision64, CleartextVector64> for Maker {
        fn synthesize_cleartext_vector(
            &mut self,
            prototype: &Self::CleartextVectorProto,
        ) -> CleartextVector64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_cleartext_vector(
            &mut self,
            entity: &CleartextVector64,
        ) -> Self::CleartextVectorProto {
            ProtoCleartextVector64(entity.to_owned())
        }
        fn destroy_cleartext_vector(&mut self, entity: CleartextVector64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
