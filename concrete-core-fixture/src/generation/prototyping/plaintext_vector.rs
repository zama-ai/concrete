use crate::generation::prototypes::{
    PlaintextVectorPrototype, ProtoPlaintextVector32, ProtoPlaintextVector64,
};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_core::prelude::{PlaintextVectorCreationEngine, PlaintextVectorRetrievalEngine};

/// A trait allowing to manipulate plaintext vector prototypes.
pub trait PrototypesPlaintextVector<Precision: IntegerPrecision> {
    type PlaintextVectorProto: PlaintextVectorPrototype<Precision = Precision>;
    fn transform_raw_vec_to_plaintext_vector(
        &mut self,
        raw: &[Precision::Raw],
    ) -> Self::PlaintextVectorProto;
    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorProto,
    ) -> Vec<Precision::Raw>;
}

impl PrototypesPlaintextVector<Precision32> for Maker {
    type PlaintextVectorProto = ProtoPlaintextVector32;

    fn transform_raw_vec_to_plaintext_vector(&mut self, raw: &[u32]) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector32(self.core_engine.create_plaintext_vector(raw).unwrap())
    }

    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorProto,
    ) -> Vec<u32> {
        self.core_engine
            .retrieve_plaintext_vector(&plaintext.0)
            .unwrap()
    }
}

impl PrototypesPlaintextVector<Precision64> for Maker {
    type PlaintextVectorProto = ProtoPlaintextVector64;

    fn transform_raw_vec_to_plaintext_vector(&mut self, raw: &[u64]) -> Self::PlaintextVectorProto {
        ProtoPlaintextVector64(self.core_engine.create_plaintext_vector(raw).unwrap())
    }

    fn transform_plaintext_vector_to_raw_vec(
        &mut self,
        plaintext: &Self::PlaintextVectorProto,
    ) -> Vec<u64> {
        self.core_engine
            .retrieve_plaintext_vector(&plaintext.0)
            .unwrap()
    }
}
