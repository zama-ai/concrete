use crate::generation::prototypes::{PlaintextPrototype, ProtoPlaintext32, ProtoPlaintext64};
use crate::generation::{IntegerPrecision, Maker, Precision32, Precision64};
use concrete_core::prelude::{PlaintextCreationEngine, PlaintextRetrievalEngine};

/// A trait allowing to manipulate plaintext prototypes.
pub trait PrototypesPlaintext<Precision: IntegerPrecision> {
    type PlaintextProto: PlaintextPrototype<Precision = Precision>;
    fn transform_raw_to_plaintext(&mut self, raw: &Precision::Raw) -> Self::PlaintextProto;
    fn transform_plaintext_to_raw(&mut self, plaintext: &Self::PlaintextProto) -> Precision::Raw;
}

impl PrototypesPlaintext<Precision32> for Maker {
    type PlaintextProto = ProtoPlaintext32;

    fn transform_raw_to_plaintext(&mut self, raw: &u32) -> Self::PlaintextProto {
        ProtoPlaintext32(self.core_engine.create_plaintext(raw).unwrap())
    }

    fn transform_plaintext_to_raw(&mut self, plaintext: &Self::PlaintextProto) -> u32 {
        self.core_engine.retrieve_plaintext(&plaintext.0).unwrap()
    }
}

impl PrototypesPlaintext<Precision64> for Maker {
    type PlaintextProto = ProtoPlaintext64;

    fn transform_raw_to_plaintext(&mut self, raw: &u64) -> Self::PlaintextProto {
        ProtoPlaintext64(self.core_engine.create_plaintext(raw).unwrap())
    }

    fn transform_plaintext_to_raw(&mut self, plaintext: &Self::PlaintextProto) -> u64 {
        self.core_engine.retrieve_plaintext(&plaintext.0).unwrap()
    }
}
