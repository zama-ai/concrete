use crate::generation::prototyping::PrototypesGgswCiphertext;
use crate::generation::IntegerPrecision;
use concrete_core::prelude::GgswCiphertextEntity;

/// A trait allowing to synthesize an actual ggsw ciphertext entity from a prototype.
pub trait SynthesizesGgswCiphertext<Precision: IntegerPrecision, GgswCiphertext>:
    PrototypesGgswCiphertext<Precision, GgswCiphertext::KeyDistribution>
where
    GgswCiphertext: GgswCiphertextEntity,
{
    fn synthesize_ggsw_ciphertext(
        &mut self,
        prototype: &Self::GgswCiphertextProto,
    ) -> GgswCiphertext;
    fn unsynthesize_ggsw_ciphertext(
        &mut self,
        entity: &GgswCiphertext,
    ) -> Self::GgswCiphertextProto;
    fn destroy_ggsw_ciphertext(&mut self, entity: GgswCiphertext);
}

#[cfg(feature = "backend_core")]
mod backend_core {
    use crate::generation::prototypes::{ProtoBinaryGgswCiphertext32, ProtoBinaryGgswCiphertext64};
    use crate::generation::synthesizing::SynthesizesGgswCiphertext;
    use crate::generation::{Maker, Precision32, Precision64};
    use concrete_core::prelude::{
        DestructionEngine, FourierGgswCiphertext32, FourierGgswCiphertext64, GgswCiphertext32,
        GgswCiphertext64, GgswCiphertextConversionEngine,
    };

    impl SynthesizesGgswCiphertext<Precision32, GgswCiphertext32> for Maker {
        fn synthesize_ggsw_ciphertext(
            &mut self,
            prototype: &Self::GgswCiphertextProto,
        ) -> GgswCiphertext32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_ggsw_ciphertext(
            &mut self,
            entity: &GgswCiphertext32,
        ) -> Self::GgswCiphertextProto {
            ProtoBinaryGgswCiphertext32(entity.to_owned())
        }

        fn destroy_ggsw_ciphertext(&mut self, entity: GgswCiphertext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGgswCiphertext<Precision64, GgswCiphertext64> for Maker {
        fn synthesize_ggsw_ciphertext(
            &mut self,
            prototype: &Self::GgswCiphertextProto,
        ) -> GgswCiphertext64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_ggsw_ciphertext(
            &mut self,
            entity: &GgswCiphertext64,
        ) -> Self::GgswCiphertextProto {
            ProtoBinaryGgswCiphertext64(entity.to_owned())
        }

        fn destroy_ggsw_ciphertext(&mut self, entity: GgswCiphertext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGgswCiphertext<Precision32, FourierGgswCiphertext32> for Maker {
        fn synthesize_ggsw_ciphertext(
            &mut self,
            prototype: &Self::GgswCiphertextProto,
        ) -> FourierGgswCiphertext32 {
            self.core_engine
                .convert_ggsw_ciphertext(&prototype.0)
                .unwrap()
        }

        fn unsynthesize_ggsw_ciphertext(
            &mut self,
            _entity: &FourierGgswCiphertext32,
        ) -> Self::GgswCiphertextProto {
            // FIXME:
            unimplemented!("The backward fourier conversion was not yet implemented");
        }

        fn destroy_ggsw_ciphertext(&mut self, entity: FourierGgswCiphertext32) {
            self.core_engine.destroy(entity).unwrap();
        }
    }

    impl SynthesizesGgswCiphertext<Precision64, FourierGgswCiphertext64> for Maker {
        fn synthesize_ggsw_ciphertext(
            &mut self,
            prototype: &Self::GgswCiphertextProto,
        ) -> FourierGgswCiphertext64 {
            self.core_engine
                .convert_ggsw_ciphertext(&prototype.0)
                .unwrap()
        }

        fn unsynthesize_ggsw_ciphertext(
            &mut self,
            _entity: &FourierGgswCiphertext64,
        ) -> Self::GgswCiphertextProto {
            // FIXME:
            unimplemented!("The backward fourier conversion was not yet implemented");
        }

        fn destroy_ggsw_ciphertext(&mut self, entity: FourierGgswCiphertext64) {
            self.core_engine.destroy(entity).unwrap();
        }
    }
}
