use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingDecryptionEngine, LweCiphertextVectorDiscardingDecryptionError,
};

/// # Description:
/// Implementation of [`LweCiphertextVectorDiscardingDecryptionEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl
    LweCiphertextVectorDiscardingDecryptionEngine<
        LweSecretKey32,
        LweCiphertextVector32,
        PlaintextVector32,
    > for CoreEngine
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
    /// let mut plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector32 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// engine.discard_decrypt_lwe_ciphertext_vector(
    ///     &key,
    ///     &mut plaintext_vector,
    ///     &ciphertext_vector,
    /// )?;
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(18));
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        output: &mut PlaintextVector32,
        input: &LweCiphertextVector32,
    ) -> Result<(), LweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        LweCiphertextVectorDiscardingDecryptionError::perform_generic_checks(key, output, input)?;
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

/// # Description:
/// Implementation of [`LweCiphertextVectorDiscardingDecryptionEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl
    LweCiphertextVectorDiscardingDecryptionEngine<
        LweSecretKey64,
        LweCiphertextVector64,
        PlaintextVector64,
    > for CoreEngine
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
    /// let mut plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector64 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// engine.discard_decrypt_lwe_ciphertext_vector(
    ///     &key,
    ///     &mut plaintext_vector,
    ///     &ciphertext_vector,
    /// )?;
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(18));
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        output: &mut PlaintextVector64,
        input: &LweCiphertextVector64,
    ) -> Result<(), LweCiphertextVectorDiscardingDecryptionError<Self::EngineError>> {
        LweCiphertextVectorDiscardingDecryptionError::perform_generic_checks(key, output, input)?;
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
