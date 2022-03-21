use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierGgswCiphertext32, FourierGgswCiphertext64, GlweCiphertext32, GlweCiphertext64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::backends::core::private::math::fft::ALLOWED_POLY_SIZE;
use crate::prelude::{CoreError, GgswCiphertextEntity};
use crate::specification::engines::{
    GlweCiphertextGgswCiphertextExternalProductEngine,
    GlweCiphertextGgswCiphertextExternalProductError,
};
use crate::specification::entities::GlweCiphertextEntity;

impl From<CoreError> for GlweCiphertextGgswCiphertextExternalProductError<CoreError> {
    fn from(err: CoreError) -> Self {
        Self::Engine(err)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextExternalProductEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl
    GlweCiphertextGgswCiphertextExternalProductEngine<
        GlweCiphertext32,
        FourierGgswCiphertext32,
        GlweCiphertext32,
    > for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize, DecompositionBaseLog, DecompositionLevelCount};
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
    /// let input_ggsw = 3_u32 << 20;
    /// let input_glwe = vec![3_u32 << 20; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_ggsw = engine.create_plaintext(&input_ggsw)?;
    /// let plaintext_glwe = engine.create_plaintext_vector(&input_glwe)?;
    ///
    /// let ggsw = engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext_ggsw, noise, level, base_log)?;
    /// let complex_ggsw: FourierGgswCiphertext32 = engine.convert_ggsw_ciphertext(&ggsw)?;
    /// let glwe = engine.encrypt_glwe_ciphertext(&key, &plaintext_glwe, noise)?;
    ///
    /// // Compute the external product.
    /// let product = engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw)?;
    /// #
    /// assert_eq!(
    /// #     product.polynomial_size(),
    /// #     glwe.polynomial_size(),
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_ggsw)?;
    /// engine.destroy(plaintext_glwe)?;
    /// engine.destroy(ggsw)?;
    /// engine.destroy(complex_ggsw)?;
    /// engine.destroy(glwe)?;
    /// engine.destroy(product)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FourierGgswCiphertext32,
    ) -> Result<GlweCiphertext32, GlweCiphertextGgswCiphertextExternalProductError<Self::EngineError>>
    {
        if !ALLOWED_POLY_SIZE.contains(&glwe_input.polynomial_size().0) {
            return Err(GlweCiphertextGgswCiphertextExternalProductError::from(
                CoreError::UnsupportedPolynomialSize,
            ));
        }
        GlweCiphertextGgswCiphertextExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input,
        )?;
        Ok(unsafe {
            self.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input,
            )
        })
    }

    unsafe fn compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FourierGgswCiphertext32,
    ) -> GlweCiphertext32 {
        let mut output = ImplGlweCiphertext::allocate(
            0u32,
            glwe_input.polynomial_size(),
            glwe_input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u32_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input
            .0
            .external_product(&mut output, &glwe_input.0, buffers);
        GlweCiphertext32(output)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextExternalProductEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl
    GlweCiphertextGgswCiphertextExternalProductEngine<
        GlweCiphertext64,
        FourierGgswCiphertext64,
        GlweCiphertext64,
    > for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize, DecompositionBaseLog, DecompositionLevelCount};
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
    /// let input_ggsw = 3_u64 << 50;
    /// let input_glwe = vec![3_u64 << 50; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_ggsw = engine.create_plaintext(&input_ggsw)?;
    /// let plaintext_glwe = engine.create_plaintext_vector(&input_glwe)?;
    ///
    /// let ggsw = engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext_ggsw, noise, level, base_log)?;
    /// let complex_ggsw: FourierGgswCiphertext64 = engine.convert_ggsw_ciphertext(&ggsw)?;
    /// let glwe = engine.encrypt_glwe_ciphertext(&key, &plaintext_glwe, noise)?;
    ///
    /// // Compute the external product.
    /// let product = engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw)?;
    /// #
    /// assert_eq!(
    /// #     product.polynomial_size(),
    /// #     glwe.polynomial_size(),
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_ggsw)?;
    /// engine.destroy(plaintext_glwe)?;
    /// engine.destroy(ggsw)?;
    /// engine.destroy(complex_ggsw)?;
    /// engine.destroy(glwe)?;
    /// engine.destroy(product)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FourierGgswCiphertext64,
    ) -> Result<GlweCiphertext64, GlweCiphertextGgswCiphertextExternalProductError<Self::EngineError>>
    {
        if !ALLOWED_POLY_SIZE.contains(&glwe_input.polynomial_size().0) {
            return Err(GlweCiphertextGgswCiphertextExternalProductError::from(
                CoreError::UnsupportedPolynomialSize,
            ));
        }
        GlweCiphertextGgswCiphertextExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input,
        )?;
        Ok(unsafe {
            self.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input,
            )
        })
    }

    unsafe fn compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FourierGgswCiphertext64,
    ) -> GlweCiphertext64 {
        let mut output = ImplGlweCiphertext::allocate(
            0u64,
            glwe_input.polynomial_size(),
            glwe_input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u64_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input
            .0
            .external_product(&mut output, &glwe_input.0, buffers);
        GlweCiphertext64(output)
    }
}
