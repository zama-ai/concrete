use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::engines::{
    GlweCiphertextEncryptionEngine, GlweCiphertextEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

impl GlweCiphertextEncryptionEngine<GlweSecretKey32, PlaintextVector32, GlweCiphertext32>
    for CoreEngine
{
    fn encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<GlweCiphertext32, GlweCiphertextEncryptionError<Self::EngineError>> {
        if key.0.polynomial_size().0 != input.0.count().0 {
            return Err(GlweCiphertextEncryptionError::PlaintextCountMismatch);
        }
        Ok(unsafe { self.encrypt_glwe_ciphertext_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> GlweCiphertext32 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0.encrypt_glwe(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertext32(ciphertext)
    }
}

impl GlweCiphertextEncryptionEngine<GlweSecretKey64, PlaintextVector64, GlweCiphertext64>
    for CoreEngine
{
    fn encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<GlweCiphertext64, GlweCiphertextEncryptionError<Self::EngineError>> {
        if key.0.polynomial_size().0 != input.0.count().0 {
            return Err(GlweCiphertextEncryptionError::PlaintextCountMismatch);
        }
        Ok(unsafe { self.encrypt_glwe_ciphertext_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> GlweCiphertext64 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0.encrypt_glwe(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertext64(ciphertext)
    }
}
