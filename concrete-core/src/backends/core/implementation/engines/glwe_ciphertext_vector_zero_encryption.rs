use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{CiphertextCount, GlweCiphertextCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertextVector32, GlweCiphertextVector64, GlweSecretKey32, GlweSecretKey64,
};
use crate::backends::core::private::crypto::glwe::GlweList as ImplGlweList;
use crate::specification::engines::{
    GlweCiphertextVectorZeroEncryptionEngine, GlweCiphertextVectorZeroEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

impl GlweCiphertextVectorZeroEncryptionEngine<GlweSecretKey32, GlweCiphertextVector32>
    for CoreEngine
{
    fn zero_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> Result<GlweCiphertextVector32, GlweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        if count.0 == 0 {
            return Err(GlweCiphertextVectorZeroEncryptionError::NullCiphertextCount);
        }
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> GlweCiphertextVector32 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(count.0),
        );
        key.0.encrypt_zero_glwe_list(
            &mut ciphertext_vector,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector32(ciphertext_vector)
    }
}

impl GlweCiphertextVectorZeroEncryptionEngine<GlweSecretKey64, GlweCiphertextVector64>
    for CoreEngine
{
    fn zero_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> Result<GlweCiphertextVector64, GlweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        if count.0 == 0 {
            return Err(GlweCiphertextVectorZeroEncryptionError::NullCiphertextCount);
        }
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> GlweCiphertextVector64 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(count.0),
        );
        key.0.encrypt_zero_glwe_list(
            &mut ciphertext_vector,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector64(ciphertext_vector)
    }
}
