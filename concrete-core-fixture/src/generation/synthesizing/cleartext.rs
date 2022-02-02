use crate::generation::prototyping::PrototypesCleartext;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::CleartextEntity;

/// A trait allowing to synthesize an actual cleartext entity from a prototype.
pub trait SynthesizesCleartext<Precision: IntegerPrecision, Cleartext>:
    PrototypesCleartext<Precision>
where
    Cleartext: CleartextEntity,
{
    fn synthesize_cleartext(&mut self, prototype: &Self::CleartextProto) -> Cleartext;
    fn unsynthesize_cleartext(&mut self, entity: &Cleartext) -> Self::CleartextProto;
    fn destroy_cleartext(&mut self, entity: Cleartext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoCleartext32, ProtoCleartext64};
    use crate::generation::synthesizing::SynthesizesCleartext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{Cleartext32, Cleartext64, DestructionEngine};

    impl SynthesizesCleartext<Precision32, Cleartext32> for Maker {
        fn synthesize_cleartext(&mut self, prototype: &Self::CleartextProto) -> Cleartext32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_cleartext(&mut self, entity: &Cleartext32) -> Self::CleartextProto {
            ProtoCleartext32(entity.to_owned())
        }

        fn destroy_cleartext(&mut self, entity: Cleartext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesCleartext<Precision64, Cleartext64> for Maker {
        fn synthesize_cleartext(&mut self, prototype: &Self::CleartextProto) -> Cleartext64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_cleartext(&mut self, entity: &Cleartext64) -> Self::CleartextProto {
            ProtoCleartext64(entity.to_owned())
        }

        fn destroy_cleartext(&mut self, entity: Cleartext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
