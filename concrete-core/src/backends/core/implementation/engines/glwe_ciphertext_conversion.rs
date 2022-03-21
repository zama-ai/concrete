use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierGlweCiphertext32, FourierGlweCiphertext64, GlweCiphertext32, GlweCiphertext64,
};
use crate::backends::core::private::crypto::glwe::FourierGlweCiphertext;
use crate::backends::core::private::math::fft::{Complex64, ALLOWED_POLY_SIZE};
use crate::prelude::CoreError;
use crate::specification::engines::{
    GlweCiphertextConversionEngine, GlweCiphertextConversionError,
};
use crate::specification::entities::GlweCiphertextEntity;

impl From<CoreError> for GlweCiphertextConversionError<CoreError> {
    fn from(err: CoreError) -> Self {
        Self::Engine(err)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextConversionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers. It converts a GLWE ciphertext from the standard to the Fourier domain.
impl GlweCiphertextConversionEngine<GlweCiphertext32, FourierGlweCiphertext32> for CoreEngine {
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u32 << 20; 256];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input)?;
    ///
    /// // We encrypt a GLWE ciphertext in the standard domain
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// // Then we convert it to the Fourier domain.
    /// let fourier_ciphertext: FourierGlweCiphertext32 =
    ///     engine.convert_glwe_ciphertext(&ciphertext)?;
    /// #
    /// assert_eq!(fourier_ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(fourier_ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(fourier_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_glwe_ciphertext(
        &mut self,
        input: &GlweCiphertext32,
    ) -> Result<FourierGlweCiphertext32, GlweCiphertextConversionError<Self::EngineError>> {
        if !ALLOWED_POLY_SIZE.contains(&input.polynomial_size().0) {
            return Err(GlweCiphertextConversionError::from(
                CoreError::UnsupportedPolynomialSize,
            ));
        }

        Ok(unsafe { self.convert_glwe_ciphertext_unchecked(input) })
    }

    unsafe fn convert_glwe_ciphertext_unchecked(
        &mut self,
        input: &GlweCiphertext32,
    ) -> FourierGlweCiphertext32 {
        let mut output = FourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u32_buffer(
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        output.fill_with_forward_fourier(&input.0, buffers);
        FourierGlweCiphertext32(output)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextConversionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers. It converts a GLWE ciphertext from the standard to the Fourier domain.
impl GlweCiphertextConversionEngine<GlweCiphertext64, FourierGlweCiphertext64> for CoreEngine {
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
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = vec![3_u64 << 50; 256];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input)?;
    ///
    /// // We encrypt a GLWE ciphertext in the standard domain
    /// let ciphertext = engine.encrypt_glwe_ciphertext(&key, &plaintext_vector, noise)?;
    ///
    /// // Then we convert it to the Fourier domain.
    /// let fourier_ciphertext: FourierGlweCiphertext64 =
    ///     engine.convert_glwe_ciphertext(&ciphertext)?;
    /// #
    /// assert_eq!(fourier_ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(fourier_ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(fourier_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn convert_glwe_ciphertext(
        &mut self,
        input: &GlweCiphertext64,
    ) -> Result<FourierGlweCiphertext64, GlweCiphertextConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_glwe_ciphertext_unchecked(input) })
    }

    unsafe fn convert_glwe_ciphertext_unchecked(
        &mut self,
        input: &GlweCiphertext64,
    ) -> FourierGlweCiphertext64 {
        let mut output = FourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u64_buffer(
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        output.fill_with_forward_fourier(&input.0, buffers);
        FourierGlweCiphertext64(output)
    }
}

/// This blanket implementation allows to convert from a type to itself by just cloning the value.
impl<Ciphertext> GlweCiphertextConversionEngine<Ciphertext, Ciphertext> for CoreEngine
where
    Ciphertext: GlweCiphertextEntity + Clone,
{
    fn convert_glwe_ciphertext(
        &mut self,
        input: &Ciphertext,
    ) -> Result<Ciphertext, GlweCiphertextConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_glwe_ciphertext_unchecked(input) })
    }

    unsafe fn convert_glwe_ciphertext_unchecked(&mut self, input: &Ciphertext) -> Ciphertext {
        (*input).clone()
    }
}
