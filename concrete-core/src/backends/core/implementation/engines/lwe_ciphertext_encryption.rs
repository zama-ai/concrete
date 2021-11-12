use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64, Plaintext32, Plaintext64,
};
use crate::backends::core::private::crypto::lwe::LweCiphertext as ImplLweCiphertext;
use crate::specification::engines::{LweCiphertextEncryptionEngine, LweCiphertextEncryptionError};
use crate::specification::entities::LweSecretKeyEntity;

impl LweCiphertextEncryptionEngine<LweSecretKey32, Plaintext32, LweCiphertext32> for CoreEngine {
    fn encrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        input: &Plaintext32,
        noise: Variance,
    ) -> Result<LweCiphertext32, LweCiphertextEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &Plaintext32,
        noise: Variance,
    ) -> LweCiphertext32 {
        let mut ciphertext = ImplLweCiphertext::allocate(0u32, key.lwe_dimension().to_lwe_size());
        key.0.encrypt_lwe(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertext32(ciphertext)
    }
}

impl LweCiphertextEncryptionEngine<LweSecretKey64, Plaintext64, LweCiphertext64> for CoreEngine {
    fn encrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        input: &Plaintext64,
        noise: Variance,
    ) -> Result<LweCiphertext64, LweCiphertextEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &Plaintext64,
        noise: Variance,
    ) -> LweCiphertext64 {
        let mut ciphertext = ImplLweCiphertext::allocate(0u64, key.lwe_dimension().to_lwe_size());
        key.0.encrypt_lwe(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertext64(ciphertext)
    }
}
