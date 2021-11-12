use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64, Plaintext32, Plaintext64,
};
use crate::backends::core::private::crypto::encoding::Plaintext as ImplPlaintext;
use crate::specification::engines::{LweCiphertextDecryptionEngine, LweCiphertextDecryptionError};

/// # Description:
/// Implementation of [`LweCiphertextDecryptionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl LweCiphertextDecryptionEngine<LweSecretKey32, LweCiphertext32, Plaintext32> for CoreEngine {
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
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// let decrypted_plaintext = engine.decrypt_lwe_ciphertext(&key, &ciphertext)?;
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(decrypted_plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertext32,
    ) -> Result<Plaintext32, LweCiphertextDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &LweCiphertext32,
    ) -> Plaintext32 {
        let mut plaintext = ImplPlaintext(0u32);
        key.0.decrypt_lwe(&mut plaintext, &input.0);
        Plaintext32(plaintext)
    }
}

/// # Description:
/// Implementation of [`LweCiphertextDecryptionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl LweCiphertextDecryptionEngine<LweSecretKey64, LweCiphertext64, Plaintext64> for CoreEngine {
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
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// let decrypted_plaintext = engine.decrypt_lwe_ciphertext(&key, &ciphertext)?;
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(decrypted_plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertext64,
    ) -> Result<Plaintext64, LweCiphertextDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.decrypt_lwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &LweCiphertext64,
    ) -> Plaintext64 {
        let mut plaintext = ImplPlaintext(0u64);
        key.0.decrypt_lwe(&mut plaintext, &input.0);
        Plaintext64(plaintext)
    }
}
