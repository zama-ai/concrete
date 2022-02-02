use crate::generation::prototypes::{CleartextPrototype, ProtoCleartext32, ProtoCleartext64};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_core::prelude::{CleartextCreationEngine, CleartextRetrievalEngine};

/// A trait allowing to manipulate cleartext prototypes.
pub trait PrototypesCleartext<Precision: IntegerPrecision> {
    type CleartextProto: CleartextPrototype<Precision = Precision>;
    fn transform_raw_to_cleartext(&mut self, raw: &Precision::Raw) -> Self::CleartextProto;
    fn transform_cleartext_to_raw(&mut self, cleartext: &Self::CleartextProto) -> Precision::Raw;
}

impl PrototypesCleartext<Precision32> for Maker {
    type CleartextProto = ProtoCleartext32;

    fn transform_raw_to_cleartext(&mut self, raw: &u32) -> Self::CleartextProto {
        ProtoCleartext32(self.core_engine.create_cleartext(raw).unwrap())
    }

    fn transform_cleartext_to_raw(&mut self, cleartext: &Self::CleartextProto) -> u32 {
        self.core_engine.retrieve_cleartext(&cleartext.0).unwrap()
    }
}

impl PrototypesCleartext<Precision64> for Maker {
    type CleartextProto = ProtoCleartext64;

    fn transform_raw_to_cleartext(&mut self, raw: &u64) -> Self::CleartextProto {
        ProtoCleartext64(self.core_engine.create_cleartext(raw).unwrap())
    }

    fn transform_cleartext_to_raw(&mut self, cleartext: &Self::CleartextProto) -> u64 {
        self.core_engine.retrieve_cleartext(&cleartext.0).unwrap()
    }
}
