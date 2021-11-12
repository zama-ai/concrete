use concrete_commons::parameters::PlaintextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{
    GlweCiphertextDecryptionEngine, GlweCiphertextDecryptionError,
};
use crate::specification::entities::{GlweCiphertextEntity, GlweSecretKeyEntity};

impl GlweCiphertextDecryptionEngine<GlweSecretKey32, GlweCiphertext32, PlaintextVector32>
    for CoreEngine
{
    fn decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertext32,
    ) -> Result<PlaintextVector32, GlweCiphertextDecryptionError<Self::EngineError>> {
        if input.glwe_dimension() != key.glwe_dimension() {
            return Err(GlweCiphertextDecryptionError::GlweDimensionMismatch);
        }
        if input.polynomial_size() != key.polynomial_size() {
            return Err(GlweCiphertextDecryptionError::PolynomialSizeMismatch);
        }
        Ok(unsafe { self.decrypt_glwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertext32,
    ) -> PlaintextVector32 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u32, PlaintextCount(key.polynomial_size().0));
        key.0.decrypt_glwe(&mut plaintext, &input.0);
        PlaintextVector32(plaintext)
    }
}

impl GlweCiphertextDecryptionEngine<GlweSecretKey64, GlweCiphertext64, PlaintextVector64>
    for CoreEngine
{
    fn decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertext64,
    ) -> Result<PlaintextVector64, GlweCiphertextDecryptionError<Self::EngineError>> {
        if input.glwe_dimension() != key.glwe_dimension() {
            return Err(GlweCiphertextDecryptionError::GlweDimensionMismatch);
        }
        if input.polynomial_size() != key.polynomial_size() {
            return Err(GlweCiphertextDecryptionError::PolynomialSizeMismatch);
        }
        Ok(unsafe { self.decrypt_glwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertext64,
    ) -> PlaintextVector64 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u64, PlaintextCount(key.polynomial_size().0));
        key.0.decrypt_glwe(&mut plaintext, &input.0);
        PlaintextVector64(plaintext)
    }
}
