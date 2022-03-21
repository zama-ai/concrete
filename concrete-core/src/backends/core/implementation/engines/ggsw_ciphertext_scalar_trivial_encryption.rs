use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
};

use crate::backends::core::entities::{
    GgswCiphertext32, GgswCiphertext64, Plaintext32, Plaintext64,
};
use crate::backends::core::private::crypto::ggsw::StandardGgswCiphertext as ImplGgswCiphertext;
use crate::specification::engines::{
    GgswCiphertextScalarTrivialEncryptionEngine, GgswCiphertextScalarTrivialEncryptionError,
};

use crate::backends::core::engines::CoreEngine;

/// # Description:
/// Implementation of [`GgswCiphertextScalarTrivialEncryptionEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl GgswCiphertextScalarTrivialEncryptionEngine<Plaintext32, GgswCiphertext32> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// let input = 3_u32 << 20;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext32 = engine.create_plaintext(&input)?;
    /// let ciphertext: GgswCiphertext32 = engine.trivially_encrypt_scalar_ggsw_ciphertext(
    ///     polynomial_size,
    ///     glwe_dimension.to_glwe_size(),
    ///     level,
    ///     base_log,
    ///     &plaintext,
    /// )?;
    ///
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    /// assert_eq!(ciphertext.decomposition_base_log(), base_log);
    /// assert_eq!(ciphertext.decomposition_level_count(), level);
    ///
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_scalar_ggsw_ciphertext(
        &mut self,
        polynomial_size: PolynomialSize,
        glwe_size: GlweSize,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        input: &Plaintext32,
    ) -> Result<GgswCiphertext32, GgswCiphertextScalarTrivialEncryptionError<Self::EngineError>>
    {
        unsafe {
            Ok(self.trivially_encrypt_scalar_ggsw_ciphertext_unchecked(
                polynomial_size,
                glwe_size,
                decomposition_level_count,
                decomposition_base_log,
                input,
            ))
        }
    }

    unsafe fn trivially_encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        polynomial_size: PolynomialSize,
        glwe_size: GlweSize,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        input: &Plaintext32,
    ) -> GgswCiphertext32 {
        let ciphertext: ImplGgswCiphertext<Vec<u32>> = ImplGgswCiphertext::new_trivial_encryption(
            polynomial_size,
            glwe_size,
            decomposition_level_count,
            decomposition_base_log,
            &input.0,
        );
        GgswCiphertext32(ciphertext)
    }
}

/// # Description:
/// Implementation of [`GgswCiphertextScalarTrivialEncryptionEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl GgswCiphertextScalarTrivialEncryptionEngine<Plaintext64, GgswCiphertext64> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// let input = 3_u64 << 20;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext64 = engine.create_plaintext(&input)?;
    /// let ciphertext: GgswCiphertext64 = engine.trivially_encrypt_scalar_ggsw_ciphertext(
    ///     polynomial_size,
    ///     glwe_dimension.to_glwe_size(),
    ///     level,
    ///     base_log,
    ///     &plaintext,
    /// )?;
    ///
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_scalar_ggsw_ciphertext(
        &mut self,
        polynomial_size: PolynomialSize,
        glwe_size: GlweSize,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        input: &Plaintext64,
    ) -> Result<GgswCiphertext64, GgswCiphertextScalarTrivialEncryptionError<Self::EngineError>>
    {
        unsafe {
            Ok(self.trivially_encrypt_scalar_ggsw_ciphertext_unchecked(
                polynomial_size,
                glwe_size,
                decomposition_level_count,
                decomposition_base_log,
                input,
            ))
        }
    }

    unsafe fn trivially_encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        polynomial_size: PolynomialSize,
        glwe_size: GlweSize,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        input: &Plaintext64,
    ) -> GgswCiphertext64 {
        let ciphertext: ImplGgswCiphertext<Vec<u64>> = ImplGgswCiphertext::new_trivial_encryption(
            polynomial_size,
            glwe_size,
            decomposition_level_count,
            decomposition_base_log,
            &input.0,
        );
        GgswCiphertext64(ciphertext)
    }
}
