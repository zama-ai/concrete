use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    LweCiphertextDiscardingDecryptionEngine, LweCiphertextDiscardingDecryptionError,
};
use crate::specification::entities::{LweCiphertextEntity, LweSecretKeyEntity};

impl LweCiphertextDiscardingDecryptionEngine<LweSecretKey32, LweCiphertext32, Plaintext32>
    for CoreEngine
{
    fn discard_decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        output: &mut Plaintext32,
        input: &LweCiphertext32,
    ) -> Result<(), LweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        if key.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextDiscardingDecryptionError::LweDimensionMismatch);
        }
        unsafe { self.discard_decrypt_lwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey32,
        output: &mut Plaintext32,
        input: &LweCiphertext32,
    ) {
        key.0.decrypt_lwe(&mut output.0, &input.0);
    }
}

impl LweCiphertextDiscardingDecryptionEngine<LweSecretKey64, LweCiphertext64, Plaintext64>
    for CoreEngine
{
    fn discard_decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        output: &mut Plaintext64,
        input: &LweCiphertext64,
    ) -> Result<(), LweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        if key.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextDiscardingDecryptionError::LweDimensionMismatch);
        }
        unsafe { self.discard_decrypt_lwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey64,
        output: &mut Plaintext64,
        input: &LweCiphertext64,
    ) {
        key.0.decrypt_lwe(&mut output.0, &input.0);
    }
}
