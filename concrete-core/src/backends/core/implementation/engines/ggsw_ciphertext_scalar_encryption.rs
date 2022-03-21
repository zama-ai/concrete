use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GgswCiphertext32, GgswCiphertext64, GlweSecretKey32, GlweSecretKey64, Plaintext32, Plaintext64,
};
use crate::backends::core::private::crypto::ggsw::StandardGgswCiphertext as ImplGgswCiphertext;
use crate::specification::engines::{
    GgswCiphertextScalarEncryptionEngine, GgswCiphertextScalarEncryptionError,
};
use crate::specification::entities::GlweSecretKeyEntity;

/// # Description:
/// Implementation of [`GgswCiphertextScalarEncryptionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl GgswCiphertextScalarEncryptionEngine<GlweSecretKey32, Plaintext32, GgswCiphertext32>
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
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    ///
    /// let ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext, noise, level, base_log)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn encrypt_scalar_ggsw_ciphertext(
        &mut self,
        key: &GlweSecretKey32,
        input: &Plaintext32,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> Result<GgswCiphertext32, GgswCiphertextScalarEncryptionError<Self::EngineError>> {
        Ok(unsafe {
            self.encrypt_scalar_ggsw_ciphertext_unchecked(
                key,
                input,
                noise,
                decomposition_level_count,
                decomposition_base_log,
            )
        })
    }

    unsafe fn encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey32,
        input: &Plaintext32,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> GgswCiphertext32 {
        let mut ciphertext = ImplGgswCiphertext::allocate(
            0u32,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
            decomposition_level_count,
            decomposition_base_log,
        );
        key.0.encrypt_constant_ggsw(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GgswCiphertext32(ciphertext)
    }
}

/// # Description:
/// Implementation of [`GgswCiphertextScalarEncryptionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl GgswCiphertextScalarEncryptionEngine<GlweSecretKey64, Plaintext64, GgswCiphertext64>
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
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    ///
    /// let ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext, noise, level, base_log)?;
    /// #
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn encrypt_scalar_ggsw_ciphertext(
        &mut self,
        key: &GlweSecretKey64,
        input: &Plaintext64,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> Result<GgswCiphertext64, GgswCiphertextScalarEncryptionError<Self::EngineError>> {
        Ok(unsafe {
            self.encrypt_scalar_ggsw_ciphertext_unchecked(
                key,
                input,
                noise,
                decomposition_level_count,
                decomposition_base_log,
            )
        })
    }

    unsafe fn encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        key: &GlweSecretKey64,
        input: &Plaintext64,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> GgswCiphertext64 {
        let mut ciphertext = ImplGgswCiphertext::allocate(
            0u64,
            key.polynomial_size(),
            key.glwe_dimension().to_glwe_size(),
            decomposition_level_count,
            decomposition_base_log,
        );
        key.0.encrypt_constant_ggsw(
            &mut ciphertext,
            &input.0,
            noise,
            &mut self.encryption_generator,
        );
        GgswCiphertext64(ciphertext)
    }
}
