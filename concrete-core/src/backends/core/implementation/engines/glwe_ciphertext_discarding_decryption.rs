use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::specification::engines::{
    GlweCiphertextDiscardingDecryptionEngine, GlweCiphertextDiscardingDecryptionError,
};
use crate::specification::entities::{
    GlweCiphertextEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};

impl GlweCiphertextDiscardingDecryptionEngine<GlweSecretKey32, GlweCiphertext32, PlaintextVector32>
    for CoreEngine
{
    fn discard_decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertext32,
    ) -> Result<(), GlweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextDiscardingDecryptionError::PolynomialSizeMismatch);
        }
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextDiscardingDecryptionError::GlweDimensionMismatch);
        }
        if input.polynomial_size().0 != output.plaintext_count().0 {
            return Err(GlweCiphertextDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_glwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertext32,
    ) {
        key.0.decrypt_glwe(&mut output.0, &input.0);
    }
}

impl GlweCiphertextDiscardingDecryptionEngine<GlweSecretKey64, GlweCiphertext64, PlaintextVector64>
    for CoreEngine
{
    fn discard_decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertext64,
    ) -> Result<(), GlweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextDiscardingDecryptionError::PolynomialSizeMismatch);
        }
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextDiscardingDecryptionError::GlweDimensionMismatch);
        }
        if input.polynomial_size().0 != output.plaintext_count().0 {
            return Err(GlweCiphertextDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_glwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertext64,
    ) {
        key.0.decrypt_glwe(&mut output.0, &input.0);
    }
}
