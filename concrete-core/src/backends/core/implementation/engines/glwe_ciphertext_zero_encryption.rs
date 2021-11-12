use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::engines::{
    GlweCiphertextZeroEncryptionEngine, GlweCiphertextZeroEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

/// # Description:
/// Implementation of [`GlweCiphertextZeroEncryptionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl GlweCiphertextZeroEncryptionEngine<GlweSecretKey32, GlweCiphertext32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(1024);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// let ciphertext = engine.zero_encrypt_glwe_ciphertext(&key, noise)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
    ) -> Result<GlweCiphertext32, GlweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
    ) -> GlweCiphertext32 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0
            .encrypt_zero_glwe(&mut ciphertext, noise, &mut self.encryption_generator);
        GlweCiphertext32(ciphertext)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextZeroEncryptionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl GlweCiphertextZeroEncryptionEngine<GlweSecretKey64, GlweCiphertext64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(1024);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// let ciphertext = engine.zero_encrypt_glwe_ciphertext(&key, noise)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
    ) -> Result<GlweCiphertext64, GlweCiphertextZeroEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_unchecked(key, noise) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
    ) -> GlweCiphertext64 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
        );
        key.0
            .encrypt_zero_glwe(&mut ciphertext, noise, &mut self.encryption_generator);
        GlweCiphertext64(ciphertext)
    }
}
