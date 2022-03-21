use crate::generation::prototyping::PrototypesGlweCiphertext;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::GlweCiphertextEntity;

/// A trait allowing to synthesize an actual glwe ciphertext entity from a prototype.
pub trait SynthesizesGlweCiphertext<Precision: IntegerPrecision, GlweCiphertext>:
    PrototypesGlweCiphertext<Precision, GlweCiphertext::KeyDistribution>
where
    GlweCiphertext: GlweCiphertextEntity,
{
    fn synthesize_glwe_ciphertext(
        &mut self,
        prototype: &Self::GlweCiphertextProto,
    ) -> GlweCiphertext;
    fn unsynthesize_glwe_ciphertext(
        &mut self,
        entity: &GlweCiphertext,
    ) -> Self::GlweCiphertextProto;
    fn destroy_glwe_ciphertext(&mut self, entity: GlweCiphertext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryGlweCiphertext32, ProtoBinaryGlweCiphertext64};
    use crate::generation::synthesizing::SynthesizesGlweCiphertext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{DestructionEngine, GlweCiphertext32, GlweCiphertext64};

    impl SynthesizesGlweCiphertext<Precision32, GlweCiphertext32> for Maker {
        fn synthesize_glwe_ciphertext(
            &mut self,
            prototype: &Self::GlweCiphertextProto,
        ) -> GlweCiphertext32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_ciphertext(
            &mut self,
            entity: &GlweCiphertext32,
        ) -> Self::GlweCiphertextProto {
            ProtoBinaryGlweCiphertext32(entity.to_owned())
        }

        fn destroy_glwe_ciphertext(&mut self, entity: GlweCiphertext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGlweCiphertext<Precision64, GlweCiphertext64> for Maker {
        fn synthesize_glwe_ciphertext(
            &mut self,
            prototype: &Self::GlweCiphertextProto,
        ) -> GlweCiphertext64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_glwe_ciphertext(
            &mut self,
            entity: &GlweCiphertext64,
        ) -> Self::GlweCiphertextProto {
            ProtoBinaryGlweCiphertext64(entity.to_owned())
        }

        fn destroy_glwe_ciphertext(&mut self, entity: GlweCiphertext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
