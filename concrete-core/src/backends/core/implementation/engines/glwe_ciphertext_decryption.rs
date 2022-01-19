use concrete_commons::parameters::PlaintextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweSecretKey32, GlweSecretKey64, PlaintextVector32,
    PlaintextVector64,
};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{
    GlweCiphertextDecryptionEngine, GlweCiphertextDecryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

/// # Description:
/// Implementation of [`GlweCiphertextDecryptionEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl GlweCiphertextDecryptionEngine<GlweSecretKey32, GlweCiphertext32, PlaintextVector32>
    for CoreEngine
{
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
    /// let polynomial_size = PolynomialSize(4);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u32 << 20; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input)?;
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// let decrypted_plaintext_vector = engine.decrypt_glwe_ciphertext(&key, &ciphertext)?;
    /// #
    /// assert_eq!(
    /// #     decrypted_plaintext_vector.plaintext_count(),
    /// #     plaintext_vector.plaintext_count()
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(decrypted_plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertext32,
    ) -> Result<PlaintextVector32, GlweCiphertextDecryptionError<Self::EngineError>> {
        GlweCiphertextDecryptionError::perform_generic_checks(key, input)?;
        Ok(unsafe { self.decrypt_glwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &GlweCiphertext32,
    ) -> PlaintextVector32 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u32, PlaintextCount(key.polynomial_size().0));
        key.0.decrypt_glwe(&mut plaintext, &input.0);
        PlaintextVector32(plaintext)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextDecryptionEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl GlweCiphertextDecryptionEngine<GlweSecretKey64, GlweCiphertext64, PlaintextVector64>
    for CoreEngine
{
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
    /// let polynomial_size = PolynomialSize(4);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = vec![3_u64 << 50; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input)?;
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// let decrypted_plaintext_vector = engine.decrypt_glwe_ciphertext(&key, &ciphertext)?;
    /// #
    /// assert_eq!(
    /// #     decrypted_plaintext_vector.plaintext_count(),
    /// #     plaintext_vector.plaintext_count()
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(decrypted_plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn decrypt_glwe_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertext64,
    ) -> Result<PlaintextVector64, GlweCiphertextDecryptionError<Self::EngineError>> {
        GlweCiphertextDecryptionError::perform_generic_checks(key, input)?;
        Ok(unsafe { self.decrypt_glwe_ciphertext_unchecked(key, input) })
    }

    unsafe fn decrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &GlweCiphertext64,
    ) -> PlaintextVector64 {
        let mut plaintext =
            ImplPlaintextList::allocate(0u64, PlaintextCount(key.polynomial_size().0));
        key.0.decrypt_glwe(&mut plaintext, &input.0);
        PlaintextVector64(plaintext)
    }
}
