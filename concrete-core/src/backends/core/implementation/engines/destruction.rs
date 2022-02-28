use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    Cleartext32, Cleartext64, CleartextVector32, CleartextVector64, FourierGgswCiphertext32,
    FourierGgswCiphertext64, FourierGlweCiphertext32, FourierGlweCiphertext64,
    FourierLweBootstrapKey32, FourierLweBootstrapKey64, GgswCiphertext32, GgswCiphertext64,
    GlweCiphertext32, GlweCiphertext64, GlweCiphertextVector32, GlweCiphertextVector64,
    GlweSecretKey32, GlweSecretKey64, LweBootstrapKey32, LweBootstrapKey64, LweCiphertext32,
    LweCiphertext64, LweCiphertextVector32, LweCiphertextVector64, LweKeyswitchKey32,
    LweKeyswitchKey64, LweSecretKey32, LweSecretKey64, PackingKeyswitchKey32,
    PackingKeyswitchKey64, Plaintext32, Plaintext64, PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::math::tensor::AsMutTensor;
use crate::specification::engines::{DestructionEngine, DestructionError};

impl DestructionEngine<Cleartext32> for CoreEngine {
    fn destroy(&mut self, entity: Cleartext32) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: Cleartext32) {}
}

impl DestructionEngine<Cleartext64> for CoreEngine {
    fn destroy(&mut self, entity: Cleartext64) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: Cleartext64) {}
}

impl DestructionEngine<CleartextVector32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: CleartextVector32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: CleartextVector32) {}
}

impl DestructionEngine<CleartextVector64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: CleartextVector64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: CleartextVector64) {}
}

impl DestructionEngine<Plaintext32> for CoreEngine {
    fn destroy(&mut self, entity: Plaintext32) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: Plaintext32) {}
}

impl DestructionEngine<Plaintext64> for CoreEngine {
    fn destroy(&mut self, entity: Plaintext64) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: Plaintext64) {}
}

impl DestructionEngine<PlaintextVector32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: PlaintextVector32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: PlaintextVector32) {}
}

impl DestructionEngine<PlaintextVector64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: PlaintextVector64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: PlaintextVector64) {}
}

impl DestructionEngine<LweCiphertext32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweCiphertext32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweCiphertext32) {}
}

impl DestructionEngine<LweCiphertext64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweCiphertext64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweCiphertext64) {}
}

impl DestructionEngine<LweCiphertextVector32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweCiphertextVector32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweCiphertextVector32) {}
}

impl DestructionEngine<LweCiphertextVector64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweCiphertextVector64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweCiphertextVector64) {}
}

impl DestructionEngine<GlweCiphertext32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweCiphertext32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GlweCiphertext32) {}
}

impl DestructionEngine<GlweCiphertext64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweCiphertext64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GlweCiphertext64) {}
}

impl DestructionEngine<FourierGlweCiphertext32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierGlweCiphertext32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierGlweCiphertext32) {}
}

impl DestructionEngine<FourierGlweCiphertext64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierGlweCiphertext64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierGlweCiphertext64) {}
}

impl DestructionEngine<GlweCiphertextVector32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweCiphertextVector32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GlweCiphertextVector32) {}
}

impl DestructionEngine<GlweCiphertextVector64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweCiphertextVector64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GlweCiphertextVector64) {}
}

impl DestructionEngine<GgswCiphertext32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GgswCiphertext32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GgswCiphertext32) {}
}

impl DestructionEngine<GgswCiphertext64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GgswCiphertext64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: GgswCiphertext64) {}
}

impl DestructionEngine<FourierGgswCiphertext32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierGgswCiphertext32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierGgswCiphertext32) {}
}

impl DestructionEngine<FourierGgswCiphertext64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierGgswCiphertext64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierGgswCiphertext64) {}
}

impl DestructionEngine<LweBootstrapKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweBootstrapKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweBootstrapKey32) {}
}

impl DestructionEngine<LweBootstrapKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweBootstrapKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweBootstrapKey64) {}
}

impl DestructionEngine<FourierLweBootstrapKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierLweBootstrapKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierLweBootstrapKey32) {}
}

impl DestructionEngine<FourierLweBootstrapKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: FourierLweBootstrapKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: FourierLweBootstrapKey64) {}
}

impl DestructionEngine<LweKeyswitchKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweKeyswitchKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweKeyswitchKey32) {}
}

impl DestructionEngine<LweKeyswitchKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweKeyswitchKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: LweKeyswitchKey64) {}
}

impl DestructionEngine<LweSecretKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweSecretKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, entity: LweSecretKey32) {
        let mut entity = entity;
        entity.0.as_mut_tensor().fill_with_element(0u32);
    }
}

impl DestructionEngine<LweSecretKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: LweSecretKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, entity: LweSecretKey64) {
        let mut entity = entity;
        entity.0.as_mut_tensor().fill_with_element(0u64);
    }
}

impl DestructionEngine<GlweSecretKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweSecretKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, entity: GlweSecretKey32) {
        let mut entity = entity;
        entity.0.as_mut_tensor().fill_with_element(0u32);
    }
}

impl DestructionEngine<GlweSecretKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: GlweSecretKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, entity: GlweSecretKey64) {
        let mut entity = entity;
        entity.0.as_mut_tensor().fill_with_element(0u64);
    }
}

impl DestructionEngine<PackingKeyswitchKey32> for CoreEngine {
    fn destroy(
        &mut self,
        entity: PackingKeyswitchKey32,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: PackingKeyswitchKey32) {}
}

impl DestructionEngine<PackingKeyswitchKey64> for CoreEngine {
    fn destroy(
        &mut self,
        entity: PackingKeyswitchKey64,
    ) -> Result<(), DestructionError<Self::EngineError>> {
        unsafe { self.destroy_unchecked(entity) };
        Ok(())
    }

    unsafe fn destroy_unchecked(&mut self, _entity: PackingKeyswitchKey64) {}
}
