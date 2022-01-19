use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::specification::engines::{
    GlweCiphertextDiscardingDecryptionEngine, GlweCiphertextDiscardingDecryptionError,
};

/// # Description:
/// Implementation of [`GlweCiphertextDiscardingDecryptionEngine`] for [`CoreEngine`] that operates
/// on 32 bits integers.
impl GlweCiphertextDiscardingDecryptionEngine<GlweSecretKey32, GlweCiphertext32, PlaintextVector32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PlaintextCount, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let mut input = vec![3_u32 << 20; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let mut plaintext_vector = engine.create_plaintext_vector(&input)?;
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// engine.discard_decrypt_glwe_ciphertext(&key, &mut plaintext_vector, &ciphertext)?;
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(4));
    ///
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertext32,
    ) -> Result<(), GlweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        GlweCiphertextDiscardingDecryptionError::perform_generic_checks(key, output, input)?;
        unsafe { self.discard_decrypt_glwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        output: &mut PlaintextVector32,
        input: &GlweCiphertext32,
    ) {
        key.0.decrypt_glwe(&mut output.0, &input.0);
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextDiscardingDecryptionEngine`] for [`CoreEngine`] that operates
/// on 64 bits integers.
impl GlweCiphertextDiscardingDecryptionEngine<GlweSecretKey64, GlweCiphertext64, PlaintextVector64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PlaintextCount, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let mut input = vec![3_u64 << 50; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let mut plaintext_vector = engine.create_plaintext_vector(&input)?;
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// engine.discard_decrypt_glwe_ciphertext(&key, &mut plaintext_vector, &ciphertext)?;
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(4));
    ///
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertext64,
    ) -> Result<(), GlweCiphertextDiscardingDecryptionError<Self::EngineError>> {
        GlweCiphertextDiscardingDecryptionError::perform_generic_checks(key, output, input)?;
        unsafe { self.discard_decrypt_glwe_ciphertext_unchecked(key, output, input) };
        Ok(())
    }

    unsafe fn discard_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        output: &mut PlaintextVector64,
        input: &GlweCiphertext64,
    ) {
        key.0.decrypt_glwe(&mut output.0, &input.0);
    }
}
