use concrete_commons::dispersion::Variance;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GgswCiphertext32, GgswCiphertext64, GlweSecretKey32, GlweSecretKey64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    GgswCiphertextScalarDiscardingEncryptionEngine, GgswCiphertextScalarDiscardingEncryptionError,
};

/// # Description:
/// Implementation of [`GgswCiphertextScalarDiscardingEncryptionEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl GgswCiphertextScalarDiscardingEncryptionEngine<GlweSecretKey32, Plaintext32, GgswCiphertext32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u32 << 20;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key_1: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let mut ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key_1, &plaintext, noise, level, base_log)?;
    /// // We're going to re-encrypt the input with another secret key
    /// // For this, it is required that the second secret key uses the same GLWE dimension
    /// // and polynomial size as the first one.
    /// let key_2: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// engine.discard_encrypt_scalar_ggsw_ciphertext(&key_2, &mut ciphertext, &plaintext, noise)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(key_1)?;
    /// engine.destroy(key_2)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_encrypt_scalar_ggsw_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut GgswCiphertext32,
        input: &Plaintext32,
        noise: Variance,
    ) -> Result<(), GgswCiphertextScalarDiscardingEncryptionError<Self::EngineError>> {
        GgswCiphertextScalarDiscardingEncryptionError::perform_generic_checks(key, output)?;
        unsafe { self.discard_encrypt_scalar_ggsw_ciphertext_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut GgswCiphertext32,
        input: &Plaintext32,
        noise: Variance,
    ) {
        key.0.encrypt_constant_ggsw(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}

/// # Description:
/// Implementation of [`GgswCiphertextScalarDiscardingEncryptionEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl GgswCiphertextScalarDiscardingEncryptionEngine<GlweSecretKey64, Plaintext64, GgswCiphertext64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = 3_u64 << 50;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key_1: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let mut ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key_1, &plaintext, noise, level, base_log)?;
    /// // We're going to re-encrypt the input with another secret key
    /// // For this, it is required that the second secret key uses the same GLWE dimension
    /// // and polynomial size as the first one.
    /// let key_2: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// engine.discard_encrypt_scalar_ggsw_ciphertext(&key_2, &mut ciphertext, &plaintext, noise)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(key_1)?;
    /// engine.destroy(key_2)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_encrypt_scalar_ggsw_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut GgswCiphertext64,
        input: &Plaintext64,
        noise: Variance,
    ) -> Result<(), GgswCiphertextScalarDiscardingEncryptionError<Self::EngineError>> {
        GgswCiphertextScalarDiscardingEncryptionError::perform_generic_checks(key, output)?;
        unsafe { self.discard_encrypt_scalar_ggsw_ciphertext_unchecked(key, output, input, noise) };
        Ok(())
    }

    unsafe fn discard_encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut GgswCiphertext64,
        input: &Plaintext64,
        noise: Variance,
    ) {
        key.0.encrypt_constant_ggsw(
            &mut output.0,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
    }
}
