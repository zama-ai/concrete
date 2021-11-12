use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingDecryptionEngine, LweCiphertextVectorDiscardingDecryptionError,
};
use crate::specification::entities::{
    LweCiphertextVectorEntity, LweSecretKeyEntity, PlaintextVectorEntity,
};

impl
    LweCiphertextVectorDiscardingDecryptionEngine<
        LweSecretKey32,
        LweCiphertextVector32,
        PlaintextVector32,
    > for CoreEngine
{
    fn discard_decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        output: &mut PlaintextVector32,
        input: &LweCiphertextVector32,
    ) -> Result<(), LweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        if key.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextVectorDiscardingDecryptionError::LweDimensionMismatch);
        }
        if input.lwe_ciphertext_count().0 != output.plaintext_count().0 {
            return Err(LweCiphertextVectorDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_lwe_ciphertext_vector_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        output: &mut PlaintextVector32,
        input: &LweCiphertextVector32,
    ) {
        key.0.decrypt_lwe_list(&mut output.0, &input.0);
    }
}

impl
    LweCiphertextVectorDiscardingDecryptionEngine<
        LweSecretKey64,
        LweCiphertextVector64,
        PlaintextVector64,
    > for CoreEngine
{
    fn discard_decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        output: &mut PlaintextVector64,
        input: &LweCiphertextVector64,
    ) -> Result<(), LweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        if key.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextVectorDiscardingDecryptionError::LweDimensionMismatch);
        }
        if input.lwe_ciphertext_count().0 != output.plaintext_count().0 {
            return Err(LweCiphertextVectorDiscardingDecryptionError::PlaintextCountMismatch);
        }
        unsafe { self.discard_decrypt_lwe_ciphertext_vector_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        output: &mut PlaintextVector64,
        input: &LweCiphertextVector64,
    ) {
        key.0.decrypt_lwe_list(&mut output.0, &input.0);
    }
}
