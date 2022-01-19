use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{CiphertextCount, GlweCiphertextCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertextVector32, GlweCiphertextVector64, GlweSecretKey32, GlweSecretKey64,
};
use crate::backends::core::private::crypto::glwe::GlweList as ImplGlweList;
use crate::specification::engines::{
    GlweCiphertextVectorZeroEncryptionEngine, GlweCiphertextVectorZeroEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

/// # Description:
/// Implementation of [`GlweCiphertextVectorZeroEncryptionEngine`] for [`CoreEngine`] that operates
/// on 32 bits integers.
impl GlweCiphertextVectorZeroEncryptionEngine<GlweSecretKey32, GlweCiphertextVector32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(1024);
    /// let ciphertext_count = GlweCiphertextCount(3);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// let ciphertext_vector =
    ///     engine.zero_encrypt_glwe_ciphertext_vector(&key, noise, ciphertext_count)?;
    /// #
    /// assert_eq!(ciphertext_vector.glwe_ciphertext_count(), ciphertext_count);
    /// assert_eq!(ciphertext_vector.polynomial_size(), polynomial_size);
    /// assert_eq!(ciphertext_vector.glwe_dimension(), glwe_dimension);
    ///
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> Result<GlweCiphertextVector32, GlweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        GlweCiphertextVectorZeroEncryptionError::perform_generic_checks(count)?;
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> GlweCiphertextVector32 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(count.0),
        );
        key.0.encrypt_zero_glwe_list(
            &mut ciphertext_vector,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector32(ciphertext_vector)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextVectorZeroEncryptionEngine`] for [`CoreEngine`] that operates
/// on 64 bits integers.
impl GlweCiphertextVectorZeroEncryptionEngine<GlweSecretKey64, GlweCiphertextVector64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweCiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(1024);
    /// let ciphertext_count = GlweCiphertextCount(3);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    ///
    /// let ciphertext_vector =
    ///     engine.zero_encrypt_glwe_ciphertext_vector(&key, noise, ciphertext_count)?;
    /// #
    /// assert_eq!(ciphertext_vector.glwe_ciphertext_count(), ciphertext_count);
    /// assert_eq!(ciphertext_vector.polynomial_size(), polynomial_size);
    /// assert_eq!(ciphertext_vector.glwe_dimension(), glwe_dimension);
    ///
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn zero_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> Result<GlweCiphertextVector64, GlweCiphertextVectorZeroEncryptionError<Self::EngineError>>
    {
        GlweCiphertextVectorZeroEncryptionError::perform_generic_checks(count)?;
        Ok(unsafe { self.zero_encrypt_glwe_ciphertext_vector_unchecked(key, noise, count) })
    }

    unsafe fn zero_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> GlweCiphertextVector64 {
        let mut ciphertext_vector = ImplGlweList::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension(),
            CiphertextCount(count.0),
        );
        key.0.encrypt_zero_glwe_list(
            &mut ciphertext_vector,
            noise,
            &mut self.encryption_generator,
        );
        GlweCiphertextVector64(ciphertext_vector)
    }
}
