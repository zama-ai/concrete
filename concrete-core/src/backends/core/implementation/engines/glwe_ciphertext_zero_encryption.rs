use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::engines::{
    GlweCiphertextZeroEncryptionEngine, GlweCiphertextZeroEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

impl GlweCiphertextZeroEncryptionEngine<GlweSecretKey32, GlweCiphertext32> for CoreEngine {
    fn zero_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
    ) -> Result<GlweCiphertext32, GlweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
    ) -> GlweCiphertext32 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0
            .encrypt_zero_glwe(&mut ciphertext, noise, &mut self.encryption_generator);
        GlweCiphertext32(ciphertext)
    }
}

impl GlweCiphertextZeroEncryptionEngine<GlweSecretKey64, GlweCiphertext64> for CoreEngine {
    fn zero_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
    ) -> Result<GlweCiphertext64, GlweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
    ) -> GlweCiphertext64 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0
            .encrypt_zero_glwe(&mut ciphertext, noise, &mut self.encryption_generator);
        GlweCiphertext64(ciphertext)
    }
}
