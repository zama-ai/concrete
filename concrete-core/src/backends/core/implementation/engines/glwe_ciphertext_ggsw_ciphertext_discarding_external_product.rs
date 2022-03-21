use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierGgswCiphertext32, FourierGgswCiphertext64, GlweCiphertext32, GlweCiphertext64,
};
use crate::backends::core::private::math::fft::ALLOWED_POLY_SIZE;
use crate::prelude::{CoreError, GgswCiphertextEntity, GlweCiphertextEntity};
use crate::specification::engines::{
    GlweCiphertextGgswCiphertextDiscardingExternalProductEngine,
    GlweCiphertextGgswCiphertextDiscardingExternalProductError,
};

impl From<CoreError> for GlweCiphertextGgswCiphertextDiscardingExternalProductError<CoreError> {
    fn from(err: CoreError) -> Self {
        Self::Engine(err)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextDiscardingExternalProductEngine`] for
/// [`CoreEngine`] that operates on 32 bits integers.
impl
    GlweCiphertextGgswCiphertextDiscardingExternalProductEngine<
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
    /// // We allocate an output ciphertext simply by cloning the input.
    /// // The content of this output ciphertext will by wiped by the external product.
    /// let mut product = glwe.clone();
    /// engine.discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw, &mut product)?;
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
    fn discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FourierGgswCiphertext32,
        output: &mut GlweCiphertext32,
    ) -> Result<(), GlweCiphertextGgswCiphertextDiscardingExternalProductError<Self::EngineError>>
    {
        if !ALLOWED_POLY_SIZE.contains(&glwe_input.polynomial_size().0) {
            return Err(
                GlweCiphertextGgswCiphertextDiscardingExternalProductError::from(
                    CoreError::UnsupportedPolynomialSize,
                ),
            );
        }
        GlweCiphertextGgswCiphertextDiscardingExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input, output,
        )?;
        unsafe {
            self.discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input, output,
            )
        };
        Ok(())
    }

    unsafe fn discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FourierGgswCiphertext32,
        output: &mut GlweCiphertext32,
    ) {
        let buffers = self.get_fourier_u32_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input
            .0
            .external_product(&mut output.0, &glwe_input.0, buffers);
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextDiscardingExternalProductEngine`] for
/// [`CoreEngine`] that operates on 64 bits integers.
impl
    GlweCiphertextGgswCiphertextDiscardingExternalProductEngine<
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
    /// // We allocate an output ciphertext simply by cloning the input.
    /// // The content of this output ciphertext will by wiped by the external product.
    /// let mut product = glwe.clone();
    /// engine.discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw, &mut product)?;
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
    fn discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FourierGgswCiphertext64,
        output: &mut GlweCiphertext64,
    ) -> Result<(), GlweCiphertextGgswCiphertextDiscardingExternalProductError<Self::EngineError>>
    {
        if !ALLOWED_POLY_SIZE.contains(&glwe_input.polynomial_size().0) {
            return Err(
                GlweCiphertextGgswCiphertextDiscardingExternalProductError::from(
                    CoreError::UnsupportedPolynomialSize,
                ),
            );
        }
        GlweCiphertextGgswCiphertextDiscardingExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input, output,
        )?;
        unsafe {
            self.discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input, output,
            )
        }
        Ok(())
    }

    unsafe fn discard_compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FourierGgswCiphertext64,
        output: &mut GlweCiphertext64,
    ) {
        let buffers = self.get_fourier_u64_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input
            .0
            .external_product(&mut output.0, &glwe_input.0, buffers);
    }
}
