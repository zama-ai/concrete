use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingEncryptionEngine, LweCiphertextVectorDiscardingEncryptionError,
};
use crate::specification::entities::{
    LweCiphertextVectorEntity, LweSecretKeyEntity, PlaintextVectorEntity,
};

impl
    LweCiphertextVectorDiscardingEncryptionEngine<
        LweSecretKey32,
        PlaintextVector32,
        LweCiphertextVector32,
    > for CoreEngine
{
    fn discard_encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        output: &mut LweCiphertextVector32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<(), LweCiphertextVectorDiscardingEncryptionError<Self::EngineError>> {
        if key.lwe_dimension() != output.lwe_dimension() {
            return Err(LweCiphertextVectorDiscardingEncryptionError::LweDimensionMismatch);
        }
        if input.plaintext_count().0 != output.lwe_ciphertext_count().0 {
            return Err(LweCiphertextVectorDiscardingEncryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_encrypt_lwe_ciphertext_vector_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        output: &mut LweCiphertextVector32,
        input: &PlaintextVector32,
        noise: Variance,
    ) {
        key.0.encrypt_lwe_list(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}

impl
    LweCiphertextVectorDiscardingEncryptionEngine<
        LweSecretKey64,
        PlaintextVector64,
        LweCiphertextVector64,
    > for CoreEngine
{
    fn discard_encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        output: &mut LweCiphertextVector64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<(), LweCiphertextVectorDiscardingEncryptionError<Self::EngineError>> {
        if key.lwe_dimension() != output.lwe_dimension() {
            return Err(LweCiphertextVectorDiscardingEncryptionError::LweDimensionMismatch);
        }
        if input.plaintext_count().0 != output.lwe_ciphertext_count().0 {
            return Err(LweCiphertextVectorDiscardingEncryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_encrypt_lwe_ciphertext_vector_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        output: &mut LweCiphertextVector64,
        input: &PlaintextVector64,
        noise: Variance,
    ) {
        key.0.encrypt_lwe_list(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}
