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

/// # Description:
/// Implementation of [`LweCiphertextVectorDecryptionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl LweCiphertextVectorDecryptionEngine<LweSecretKey32, LweCiphertextVector32, PlaintextVector32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{LweCiphertextCount, LweDimension, PlaintextCount};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(6);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u32 << 20; 18];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector32 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// let decrypted_plaintext_vector =
    ///     engine.decrypt_lwe_ciphertext_vector(&key, &ciphertext_vector)?;
    ///
    /// assert_eq!(
    ///     decrypted_plaintext_vector.plaintext_count(),
    ///     PlaintextCount(18)
    /// );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(decrypted_plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertextVector32,
    ) -> Result<PlaintextVector32, LweCiphertextVectorDecryptionError<Self::EngineError>> {
        LweCiphertextVectorDecryptionError::perform_generic_checks(key, input)?;
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

/// # Description:
/// Implementation of [`LweCiphertextVectorDecryptionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl LweCiphertextVectorDecryptionEngine<LweSecretKey64, LweCiphertextVector64, PlaintextVector64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{LweCiphertextCount, LweDimension, PlaintextCount};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(6);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = vec![3_u64 << 50; 18];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector64 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// let decrypted_plaintext_vector =
    ///     engine.decrypt_lwe_ciphertext_vector(&key, &ciphertext_vector)?;
    ///
    /// assert_eq!(
    ///     decrypted_plaintext_vector.plaintext_count(),
    ///     PlaintextCount(18)
    /// );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(decrypted_plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertextVector64,
    ) -> Result<PlaintextVector64, LweCiphertextVectorDecryptionError<Self::EngineError>> {
        LweCiphertextVectorDecryptionError::perform_generic_checks(key, input)?;
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
