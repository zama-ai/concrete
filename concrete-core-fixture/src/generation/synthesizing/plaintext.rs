use crate::generation::prototyping::PrototypesPlaintext;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::PlaintextEntity;

/// A trait allowing to synthesize an actual plaintext entity from a prototype.
pub trait SynthesizesPlaintext<Precision: IntegerPrecision, Plaintext>:
    PrototypesPlaintext<Precision>
where
    Plaintext: PlaintextEntity,
{
    fn synthesize_plaintext(&mut self, prototype: &Self::PlaintextProto) -> Plaintext;
    fn unsynthesize_plaintext(&mut self, entity: &Plaintext) -> Self::PlaintextProto;
    fn destroy_plaintext(&mut self, entity: Plaintext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoPlaintext32, ProtoPlaintext64};
    use crate::generation::synthesizing::SynthesizesPlaintext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, Plaintext32, Plaintext64};

    impl SynthesizesPlaintext<Precision32, Plaintext32> for Maker {
        fn synthesize_plaintext(&mut self, prototype: &Self::PlaintextProto) -> Plaintext32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_plaintext(&mut self, entity: &Plaintext32) -> Self::PlaintextProto {
            ProtoPlaintext32(entity.to_owned())
        }

        fn destroy_plaintext(&mut self, entity: Plaintext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesPlaintext<Precision64, Plaintext64> for Maker {
        fn synthesize_plaintext(&mut self, prototype: &Self::PlaintextProto) -> Plaintext64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_plaintext(&mut self, entity: &Plaintext64) -> Self::PlaintextProto {
            ProtoPlaintext64(entity.to_owned())
        }

        fn destroy_plaintext(&mut self, entity: Plaintext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
