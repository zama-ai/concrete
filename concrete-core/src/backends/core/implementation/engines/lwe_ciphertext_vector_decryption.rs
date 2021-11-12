use concrete_commons::parameters::PlaintextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{
    LweCiphertextVectorDecryptionEngine, LweCiphertextVectorDecryptionError,
};
use crate::specification::entities::LweCiphertextVectorEntity;

impl LweCiphertextVectorDecryptionEngine<LweSecretKey32, LweCiphertextVector32, PlaintextVector32>
    for CoreEngine
{
    fn decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertextVector32,
    ) -> Result<PlaintextVector32, LweCiphertextVectorDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_vector_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertextVector32,
    ) -> PlaintextVector32 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u32, PlaintextCount(input.lwe_ciphertext_count().0));
        key.0.decrypt_lwe_list(&mut plaintext, &input.0);
        PlaintextVector32(plaintext)
    }
}

impl LweCiphertextVectorDecryptionEngine<LweSecretKey64, LweCiphertextVector64, PlaintextVector64>
    for CoreEngine
{
    fn decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertextVector64,
    ) -> Result<PlaintextVector64, LweCiphertextVectorDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_vector_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertextVector64,
    ) -> PlaintextVector64 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u64, PlaintextCount(input.lwe_ciphertext_count().0));
        key.0.decrypt_lwe_list(&mut plaintext, &input.0);
        PlaintextVector64(plaintext)
    }
}
