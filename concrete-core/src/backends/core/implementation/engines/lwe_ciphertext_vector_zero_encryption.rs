use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{CiphertextCount, LweCiphertextCount, PlaintextCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::backends::core::private::crypto::lwe::LweList as ImplLweList;
use crate::specification::engines::{
    LweCiphertextVectorZeroEncryptionEngine, LweCiphertextVectorZeroEncryptionError,
};
use crate::specification::entities::LweSecretKeyEntity;

impl LweCiphertextVectorZeroEncryptionEngine<LweSecretKey32, LweCiphertextVector32> for CoreEngine {
    fn zero_encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        noise: Variance,
        count: LweCiphertextCount,
    ) -> Result<LweCiphertextVector32, LweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        if count.0 == 0 {
            return Err(LweCiphertextVectorZeroEncryptionError::NullCiphertextCount);
        }
        Ok(unsafe { self.zero_encrypt_lwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        noise: Variance,
        count: LweCiphertextCount,
    ) -> LweCiphertextVector32 {
        let mut vector = ImplLweList::allocate(
            0u32,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(count.0),
        );
        let plaintexts = ImplPlaintextList::allocate(0u32, PlaintextCount(count.0));
        key.0.encrypt_lwe_list(
            &mut vector,
            &plaintexts,
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertextVector32(vector)
    }
}

impl LweCiphertextVectorZeroEncryptionEngine<LweSecretKey64, LweCiphertextVector64> for CoreEngine {
    fn zero_encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        noise: Variance,
        count: LweCiphertextCount,
    ) -> Result<LweCiphertextVector64, LweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        if count.0 == 0 {
            return Err(LweCiphertextVectorZeroEncryptionError::NullCiphertextCount);
        }
        Ok(unsafe { self.zero_encrypt_lwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        noise: Variance,
        count: LweCiphertextCount,
    ) -> LweCiphertextVector64 {
        let mut vector = ImplLweList::allocate(
            0u64,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(count.0),
        );
        let plaintexts = ImplPlaintextList::allocate(0u64, PlaintextCount(count.0));
        key.0.encrypt_lwe_list(
            &mut vector,
            &plaintexts,
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertextVector64(vector)
    }
}
