use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::CiphertextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::lwe::LweList as ImplLweList;
use crate::specification::engines::{
    LweCiphertextVectorEncryptionEngine, LweCiphertextVectorEncryptionError,
};
use crate::specification::entities::{LweSecretKeyEntity, PlaintextVectorEntity};

impl LweCiphertextVectorEncryptionEngine<LweSecretKey32, PlaintextVector32, LweCiphertextVector32>
    for CoreEngine
{
    fn encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<LweCiphertextVector32, LweCiphertextVectorEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> LweCiphertextVector32 {
        let mut vector = ImplLweList::allocate(
            0u32,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(input.plaintext_count().0),
        );
        key.0
            .encrypt_lwe_list(&mut vector, &input.0, noise, &mut self.encryption_generator);
        LweCiphertextVector32(vector)
    }
}

impl LweCiphertextVectorEncryptionEngine<LweSecretKey64, PlaintextVector64, LweCiphertextVector64>
    for CoreEngine
{
    fn encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<LweCiphertextVector64, LweCiphertextVectorEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> LweCiphertextVector64 {
        let mut vector = ImplLweList::allocate(
            0u64,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(input.plaintext_count().0),
        );
        key.0
            .encrypt_lwe_list(&mut vector, &input.0, noise, &mut self.encryption_generator);
        LweCiphertextVector64(vector)
    }
}
