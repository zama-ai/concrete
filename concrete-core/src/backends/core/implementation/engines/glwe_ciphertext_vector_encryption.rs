use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::CiphertextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertextVector32, GlweCiphertextVector64, GlweSecretKey32, GlweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::glwe::GlweList as ImplGlweList;
use crate::specification::engines::{
    GlweCiphertextVectorEncryptionEngine, GlweCiphertextVectorEncryptionError,
};
use crate::specification::entities::{GlweSecretKeyEntity, PlaintextVectorEntity};

impl
    GlweCiphertextVectorEncryptionEngine<GlweSecretKey32, PlaintextVector32, GlweCiphertextVector32>
    for CoreEngine
{
    fn encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<GlweCiphertextVector32, GlweCiphertextVectorEncryptionError<Self::EngineError>>
    {
        if (input.plaintext_count().0 % key.polynomial_size().0) != 0 {
            return Err(GlweCiphertextVectorEncryptionError::PlaintextCountMismatch);
        }
        Ok(unsafe { self.encrypt_glwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> GlweCiphertextVector32 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(input.plaintext_count().0 / key.polynomial_size().0),
        );
        key.0.encrypt_glwe_list(
            &mut ciphertext_vector,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector32(ciphertext_vector)
    }
}

impl
    GlweCiphertextVectorEncryptionEngine<GlweSecretKey64, PlaintextVector64, GlweCiphertextVector64>
    for CoreEngine
{
    fn encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<GlweCiphertextVector64, GlweCiphertextVectorEncryptionError<Self::EngineError>>
    {
        if (input.plaintext_count().0 % key.polynomial_size().0) != 0 {
            return Err(GlweCiphertextVectorEncryptionError::PlaintextCountMismatch);
        }
        Ok(unsafe { self.encrypt_glwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> GlweCiphertextVector64 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(input.plaintext_count().0 / key.polynomial_size().0),
        );
        key.0.encrypt_glwe_list(
            &mut ciphertext_vector,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector64(ciphertext_vector)
    }
}
