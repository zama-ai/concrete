use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierGgswCiphertext32, FourierGgswCiphertext64, GgswCiphertext32, GgswCiphertext64,
};
use crate::backends::core::private::crypto::ggsw::FourierGgswCiphertext;
use crate::backends::core::private::math::fft::Complex64;
use crate::specification::engines::{
    GgswCiphertextConversionEngine, GgswCiphertextConversionError,
};
use crate::specification::entities::GgswCiphertextEntity;

/// # Description:
/// Implementation of [`GgswCiphertextConversionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers. It converts a GGSW ciphertext from the standard to the Fourier domain.
impl GgswCiphertextConversionEngine<GgswCiphertext32, FourierGgswCiphertext32> for CoreEngine {
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
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
    /// // We encrypt a GGSW ciphertext in the standard domain
    /// let ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext, noise, level, base_log)?;
    ///
    /// // Then we convert it to the Fourier domain.
    /// let fourier_ciphertext: FourierGgswCiphertext32 =
    ///     engine.convert_ggsw_ciphertext(&ciphertext)?;
    /// #
    /// assert_eq!(fourier_ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(fourier_ciphertext.polynomial_size(), polynomial_size);
    /// assert_eq!(fourier_ciphertext.decomposition_base_log(), base_log);
    /// assert_eq!(fourier_ciphertext.decomposition_level_count(), level);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(fourier_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_ggsw_ciphertext(
        &mut self,
        input: &GgswCiphertext32,
    ) -> Result<FourierGgswCiphertext32, GgswCiphertextConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_ggsw_ciphertext_unchecked(input) })
    }

    unsafe fn convert_ggsw_ciphertext_unchecked(
        &mut self,
        input: &GgswCiphertext32,
    ) -> FourierGgswCiphertext32 {
        let mut output = FourierGgswCiphertext::allocate(
            Complex64::new(0., 0.),
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
            input.decomposition_level_count(),
            input.decomposition_base_log(),
        );
        let buffers = self.get_fourier_u32_buffer(
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        output.fill_with_forward_fourier(&input.0, buffers);
        FourierGgswCiphertext32(output)
    }
}

/// # Description:
/// Implementation of [`GgswCiphertextConversionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers. It converts a GGSW ciphertext from the standard to the Fourier domain.
impl GgswCiphertextConversionEngine<GgswCiphertext64, FourierGgswCiphertext64> for CoreEngine {
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
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
    /// // We encrypt a GGSW ciphertext in the standard domain
    /// let ciphertext =
    ///     engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext, noise, level, base_log)?;
    ///
    /// // Then we convert it to the Fourier domain.
    /// let fourier_ciphertext: FourierGgswCiphertext64 =
    ///     engine.convert_ggsw_ciphertext(&ciphertext)?;
    /// #
    /// assert_eq!(fourier_ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(fourier_ciphertext.polynomial_size(), polynomial_size);
    /// assert_eq!(fourier_ciphertext.decomposition_base_log(), base_log);
    /// assert_eq!(fourier_ciphertext.decomposition_level_count(), level);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(fourier_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_ggsw_ciphertext(
        &mut self,
        input: &GgswCiphertext64,
    ) -> Result<FourierGgswCiphertext64, GgswCiphertextConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_ggsw_ciphertext_unchecked(input) })
    }

    unsafe fn convert_ggsw_ciphertext_unchecked(
        &mut self,
        input: &GgswCiphertext64,
    ) -> FourierGgswCiphertext64 {
        let mut output = FourierGgswCiphertext::allocate(
            Complex64::new(0., 0.),
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
            input.decomposition_level_count(),
            input.decomposition_base_log(),
        );
        let buffers = self.get_fourier_u64_buffer(
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        output.fill_with_forward_fourier(&input.0, buffers);
        FourierGgswCiphertext64(output)
    }
}

/// This blanket implementation allows to convert from a type to itself by just cloning the value.
impl<Ciphertext> GgswCiphertextConversionEngine<Ciphertext, Ciphertext> for CoreEngine
where
    Ciphertext: GgswCiphertextEntity + Clone,
{
    fn convert_ggsw_ciphertext(
        &mut self,
        input: &Ciphertext,
    ) -> Result<Ciphertext, GgswCiphertextConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_ggsw_ciphertext_unchecked(input) })
    }

    unsafe fn convert_ggsw_ciphertext_unchecked(&mut self, input: &Ciphertext) -> Ciphertext {
        (*input).clone()
    }
}
