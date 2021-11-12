use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertextVector32, GlweCiphertextVector64, GlweSecretKey32, GlweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::specification::engines::{
    GlweCiphertextVectorDiscardingDecryptionEngine, GlweCiphertextVectorDiscardingDecryptionError,
};
use crate::specification::entities::{
    GlweCiphertextVectorEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};

impl
    GlweCiphertextVectorDiscardingDecryptionEngine<
        GlweSecretKey32,
        GlweCiphertextVector32,
        PlaintextVector32,
    > for CoreEngine
{
    fn discard_decrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertextVector32,
    ) -> Result<(), GlweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::PolynomialSizeMismatch);
        }
        if output.plaintext_count().0
            != (input.polynomial_size().0 * input.glwe_ciphertext_count().0)
        {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_glwe_ciphertext_vector_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertextVector32,
    ) {
        key.0.decrypt_glwe_list(&mut output.0, &input.0);
    }
}

impl
    GlweCiphertextVectorDiscardingDecryptionEngine<
        GlweSecretKey64,
        GlweCiphertextVector64,
        PlaintextVector64,
    > for CoreEngine
{
    fn discard_decrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertextVector64,
    ) -> Result<(), GlweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        if key.glwe_dimension() != input.glwe_dimension() {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size() != input.polynomial_size() {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::PolynomialSizeMismatch);
        }
        if output.plaintext_count().0
            != (input.polynomial_size().0 * input.glwe_ciphertext_count().0)
        {
            return Err(GlweCiphertextVectorDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_glwe_ciphertext_vector_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertextVector64,
    ) {
        key.0.decrypt_glwe_list(&mut output.0, &input.0);
    }
}
