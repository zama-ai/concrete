use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64, Plaintext32, Plaintext64,
};
use crate::backends::core::private::crypto::encoding::Plaintext as ImplPlaintext;
use crate::specification::engines::{LweCiphertextDecryptionEngine, LweCiphertextDecryptionError};

impl LweCiphertextDecryptionEngine<LweSecretKey32, LweCiphertext32, Plaintext32> for CoreEngine {
    fn decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertext32,
    ) -> Result<Plaintext32, LweCiphertextDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertext32,
    ) -> Plaintext32 {
        let mut plaintext = ImplPlaintext(0u32);
        key.0.decrypt_lwe(&mut plaintext, &input.0);
        Plaintext32(plaintext)
    }
}

impl LweCiphertextDecryptionEngine<LweSecretKey64, LweCiphertext64, Plaintext64> for CoreEngine {
    fn decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertext64,
    ) -> Result<Plaintext64, LweCiphertextDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertext64,
    ) -> Plaintext64 {
        let mut plaintext = ImplPlaintext(0u64);
        key.0.decrypt_lwe(&mut plaintext, &input.0);
        Plaintext64(plaintext)
    }
}
