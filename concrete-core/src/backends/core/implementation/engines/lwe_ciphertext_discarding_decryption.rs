use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    LweCiphertextDiscardingDecryptionEngine, LweCiphertextDiscardingDecryptionError,
};

/// # Description:
/// Implementation of [`LweCiphertextDiscardingDecryptionEngine`] for [`CoreEngine`] that operates
/// on 32 bits integers.
impl LweCiphertextDiscardingDecryptionEngine<LweSecretKey32, LweCiphertext32, Plaintext32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(2);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u32 << 20;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let mut plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.discard_decrypt_lwe_ciphertext(&key, &mut plaintext, &ciphertext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        output: &mut Plaintext32,
        input: &LweCiphertext32,
    ) -> Result<(), LweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        LweCiphertextDiscardingDecryptionError::perform_generic_checks(key, input)?;
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

/// # Description:
/// Implementation of [`LweCiphertextDiscardingDecryptionEngine`] for [`CoreEngine`] that operates
/// on 64 bits integers.
impl LweCiphertextDiscardingDecryptionEngine<LweSecretKey64, LweCiphertext64, Plaintext64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(2);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = 3_u64 << 50;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let mut plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.discard_decrypt_lwe_ciphertext(&key, &mut plaintext, &ciphertext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        output: &mut Plaintext64,
        input: &LweCiphertext64,
    ) -> Result<(), LweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        LweCiphertextDiscardingDecryptionError::perform_generic_checks(key, input)?;
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
