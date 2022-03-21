use crate::generation::prototypes::{
    CleartextVectorPrototype, ProtoCleartextVector32, ProtoCleartextVector64,
};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_core::prelude::{CleartextVectorCreationEngine, CleartextVectorRetrievalEngine};

/// A trait allowing to manipulate cleartext vector prototypes.
pub trait PrototypesCleartextVector<Precision: IntegerPrecision> {
    type CleartextVectorProto: CleartextVectorPrototype<Precision = Precision>;
    fn transform_raw_vec_to_cleartext_vector(
        &mut self,
        raw: &[Precision::Raw],
    ) -> Self::CleartextVectorProto;
    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorProto,
    ) -> Vec<Precision::Raw>;
}

impl PrototypesCleartextVector<Precision32> for Maker {
    type CleartextVectorProto = ProtoCleartextVector32;

    fn transform_raw_vec_to_cleartext_vector(&mut self, raw: &[u32]) -> Self::CleartextVectorProto {
        ProtoCleartextVector32(self.core_engine.create_cleartext_vector(raw).unwrap())
    }

    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorProto,
    ) -> Vec<u32> {
        self.core_engine
            .retrieve_cleartext_vector(&cleartext.0)
            .unwrap()
    }
}

impl PrototypesCleartextVector<Precision64> for Maker {
    type CleartextVectorProto = ProtoCleartextVector64;

    fn transform_raw_vec_to_cleartext_vector(&mut self, raw: &[u64]) -> Self::CleartextVectorProto {
        ProtoCleartextVector64(self.core_engine.create_cleartext_vector(raw).unwrap())
    }

    fn transform_cleartext_vector_to_raw_vec(
        &mut self,
        cleartext: &Self::CleartextVectorProto,
    ) -> Vec<u64> {
        self.core_engine
            .retrieve_cleartext_vector(&cleartext.0)
            .unwrap()
    }
}
