use concrete_commons::parameters::PlaintextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertextVector32, GlweCiphertextVector64, GlweSecretKey32, GlweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{
    GlweCiphertextVectorDecryptionEngine, GlweCiphertextVectorDecryptionError,
};
use crate::specification::entities::{GlweCiphertextVectorEntity, GlweSecretKeyEntity};

impl
    GlweCiphertextVectorDecryptionEngine<GlweSecretKey32, GlweCiphertextVector32, PlaintextVector32>
    for CoreEngine
{
    fn decrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertextVector32,
    ) -> Result<PlaintextVector32, GlweCiphertextVectorDecryptionError<Self::EngineError>> {
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextVectorDecryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextVectorDecryptionError::PolynomialSizeMismatch);
        }
        Ok(unsafe { self.decrypt_glwe_ciphertext_vector_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertextVector32,
    ) -> PlaintextVector32 {
        let mut plaintext_list = ImplPlaintextList::allocate(
            0u32,
            PlaintextCount(key.polynomial_size().0 * key.glwe_dimension().0),
        );
        key.0.decrypt_glwe_list(&mut plaintext_list, &input.0);
        PlaintextVector32(plaintext_list)
    }
}

impl
    GlweCiphertextVectorDecryptionEngine<GlweSecretKey64, GlweCiphertextVector64, PlaintextVector64>
    for CoreEngine
{
    fn decrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertextVector64,
    ) -> Result<PlaintextVector64, GlweCiphertextVectorDecryptionError<Self::EngineError>> {
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextVectorDecryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextVectorDecryptionError::PolynomialSizeMismatch);
        }
        Ok(unsafe { self.decrypt_glwe_ciphertext_vector_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertextVector64,
    ) -> PlaintextVector64 {
        let mut plaintext_list = ImplPlaintextList::allocate(
            0u64,
            PlaintextCount(key.polynomial_size().0 * key.glwe_dimension().0),
        );
        key.0.decrypt_glwe_list(&mut plaintext_list, &input.0);
        PlaintextVector64(plaintext_list)
    }
}
