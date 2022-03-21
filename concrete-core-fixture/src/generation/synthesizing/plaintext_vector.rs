use crate::generation::prototyping::PrototypesPlaintextVector;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::PlaintextVectorEntity;

/// A trait allowing to synthesize an actual plaintext vector entity from a prototype.
pub trait SynthesizesPlaintextVector<Precision: IntegerPrecision, PlaintextVector>:
    PrototypesPlaintextVector<Precision>
where
    PlaintextVector: PlaintextVectorEntity,
{
    fn synthesize_plaintext_vector(
        &mut self,
        prototype: &Self::PlaintextVectorProto,
    ) -> PlaintextVector;
    fn unsynthesize_plaintext_vector(
        &mut self,
        entity: &PlaintextVector,
    ) -> Self::PlaintextVectorProto;
    fn destroy_plaintext_vector(&mut self, entity: PlaintextVector);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoPlaintextVector32, ProtoPlaintextVector64};
    use crate::generation::synthesizing::SynthesizesPlaintextVector;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, PlaintextVector32, PlaintextVector64};

    impl SynthesizesPlaintextVector<Precision32, PlaintextVector32> for Maker {
        fn synthesize_plaintext_vector(
            &mut self,
            prototype: &Self::PlaintextVectorProto,
        ) -> PlaintextVector32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_plaintext_vector(
            &mut self,
            entity: &PlaintextVector32,
        ) -> Self::PlaintextVectorProto {
            ProtoPlaintextVector32(entity.to_owned())
        }

        fn destroy_plaintext_vector(&mut self, entity: PlaintextVector32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesPlaintextVector<Precision64, PlaintextVector64> for Maker {
        fn synthesize_plaintext_vector(
            &mut self,
            prototype: &Self::PlaintextVectorProto,
        ) -> PlaintextVector64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_plaintext_vector(
            &mut self,
            entity: &PlaintextVector64,
        ) -> Self::PlaintextVectorProto {
            ProtoPlaintextVector64(entity.to_owned())
        }

        fn destroy_plaintext_vector(&mut self, entity: PlaintextVector64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
