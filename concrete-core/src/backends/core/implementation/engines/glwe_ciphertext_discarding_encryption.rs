use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::specification::engines::{
    GlweCiphertextDiscardingEncryptionEngine, GlweCiphertextDiscardingEncryptionError,
};
use crate::specification::entities::{
    GlweCiphertextEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};

impl GlweCiphertextDiscardingEncryptionEngine<GlweSecretKey32, PlaintextVector32, GlweCiphertext32>
    for CoreEngine
{
    fn discard_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut GlweCiphertext32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<(), GlweCiphertextDiscardingEncryptionError<Self::EngineError>> {
        if key.polynomial_size() != output.polynomial_size() {
            return Err(GlweCiphertextDiscardingEncryptionError::PolynomialSizeMismatch);
        }
        if key.glwe_dimension() != output.glwe_dimension() {
            return Err(GlweCiphertextDiscardingEncryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size().0 != input.plaintext_count().0 {
            return Err(GlweCiphertextDiscardingEncryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_encrypt_glwe_ciphertext_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut GlweCiphertext32,
        input: &PlaintextVector32,
        noise: Variance,
    ) {
        key.0.encrypt_glwe(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}

impl GlweCiphertextDiscardingEncryptionEngine<GlweSecretKey64, PlaintextVector64, GlweCiphertext64>
    for CoreEngine
{
    fn discard_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut GlweCiphertext64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<(), GlweCiphertextDiscardingEncryptionError<Self::EngineError>> {
        if key.polynomial_size() != output.polynomial_size() {
            return Err(GlweCiphertextDiscardingEncryptionError::PolynomialSizeMismatch);
        }
        if key.glwe_dimension() != output.glwe_dimension() {
            return Err(GlweCiphertextDiscardingEncryptionError::GlweDimensionMismatch);
        }
        if key.polynomial_size().0 != input.plaintext_count().0 {
            return Err(GlweCiphertextDiscardingEncryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_encrypt_glwe_ciphertext_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut GlweCiphertext64,
        input: &PlaintextVector64,
        noise: Variance,
    ) {
        key.0.encrypt_glwe(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}
