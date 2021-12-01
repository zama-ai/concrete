use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweSecretKey32, LweSecretKey64,
};
use crate::backends::core::private::crypto::encoding::Plaintext as ImplPlaintext;
use crate::backends::core::private::crypto::lwe::LweCiphertext as ImplLweCiphertext;
use crate::specification::engines::{
    LweCiphertextZeroEncryptionEngine, LweCiphertextZeroEncryptionError,
};
use crate::specification::entities::LweSecretKeyEntity;

/// # Description:
/// Implementation of [`LweCiphertextZeroEncryptionEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl LweCiphertextZeroEncryptionEngine<LweSecretKey32, LweCiphertext32> for CoreEngine {
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
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    ///
    /// let ciphertext = engine.zero_encrypt_lwe_ciphertext(&key, noise)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey32,
        noise: Variance,
    ) -> Result<LweCiphertext32, LweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_lwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey32,
        noise: Variance,
    ) -> LweCiphertext32 {
        let mut ciphertext = ImplLweCiphertext::allocate(0u32, key.lwe_dimension().to_lwe_size());
        key.0.encrypt_lwe(
            &mut ciphertext,
            &ImplPlaintext(0u32),
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertext32(ciphertext)
    }
}

/// # Description:
/// Implementation of [`LweCiphertextZeroEncryptionEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl LweCiphertextZeroEncryptionEngine<LweSecretKey64, LweCiphertext64> for CoreEngine {
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
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    ///
    /// let ciphertext = engine.zero_encrypt_lwe_ciphertext(&key, noise)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_lwe_ciphertext(
        &mut self,
        key: &LweSecretKey64,
        noise: Variance,
    ) -> Result<LweCiphertext64, LweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_lwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &LweSecretKey64,
        noise: Variance,
    ) -> LweCiphertext64 {
        let mut ciphertext = ImplLweCiphertext::allocate(0u64, key.lwe_dimension().to_lwe_size());
        key.0.encrypt_lwe(
            &mut ciphertext,
            &ImplPlaintext(0u64),
            noise,
            &mut self.encryption_generator,
        );
        LweCiphertext64(ciphertext)
    }
}
