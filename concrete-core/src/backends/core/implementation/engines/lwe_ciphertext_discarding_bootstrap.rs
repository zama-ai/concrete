use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierLweBootstrapKey32, FourierLweBootstrapKey64, GlweCiphertext32, GlweCiphertext64,
    LweCiphertext32, LweCiphertext64,
};
use crate::backends::core::private::math::fft::ALLOWED_POLY_SIZE;
use crate::prelude::{CoreError, GlweCiphertextEntity, LweBootstrapKeyEntity};
use crate::specification::engines::{
    LweCiphertextDiscardingBootstrapEngine, LweCiphertextDiscardingBootstrapError,
};

impl From<CoreError> for LweCiphertextDiscardingBootstrapError<CoreError> {
    fn from(err: CoreError) -> Self {
        Self::Engine(err)
    }
}

/// # Description:
/// Implementation of [`LweCiphertextDiscardingBootstrapEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl
    LweCiphertextDiscardingBootstrapEngine<
        FourierLweBootstrapKey32,
        GlweCiphertext32,
        LweCiphertext32,
        LweCiphertext32,
    > for CoreEngine
{
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
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u32 << 20;
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, lwe_dim_output, glwe_dim, poly_size) = (
    ///     LweDimension(4),
    ///     LweDimension(1024),
    ///     GlweDimension(1),
    ///     PolynomialSize(1024),
    /// );
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// // A constant function is applied during the bootstrap
    /// let lut = vec![8_u32 << 20; poly_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    /// let bsk: FourierLweBootstrapKey32 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// let lwe_sk_output: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dim_output)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&lut)?;
    /// let acc =
    ///     engine.trivially_encrypt_glwe_ciphertext(glwe_dim.to_glwe_size(), &plaintext_vector)?;
    /// let input = engine.encrypt_lwe_ciphertext(&lwe_sk, &plaintext, noise)?;
    /// let mut output = engine.zero_encrypt_lwe_ciphertext(&lwe_sk_output, noise)?;
    ///
    /// engine.discard_bootstrap_lwe_ciphertext(&mut output, &input, &acc, &bsk)?;
    /// #
    /// assert_eq!(output.lwe_dimension(), lwe_dim_output);
    ///
    /// engine.destroy(lwe_sk)?;
    /// engine.destroy(glwe_sk)?;
    /// engine.destroy(bsk)?;
    /// engine.destroy(lwe_sk_output)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(acc)?;
    /// engine.destroy(input)?;
    /// engine.destroy(output)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
        acc: &GlweCiphertext32,
        bsk: &FourierLweBootstrapKey32,
    ) -> Result<(), LweCiphertextDiscardingBootstrapError<Self::EngineError>> {
        if !ALLOWED_POLY_SIZE.contains(&acc.polynomial_size().0) {
            return Err(LweCiphertextDiscardingBootstrapError::from(
                CoreError::UnsupportedPolynomialSize,
            ));
        }
        LweCiphertextDiscardingBootstrapError::perform_generic_checks(output, input, acc, bsk)?;
        unsafe { self.discard_bootstrap_lwe_ciphertext_unchecked(output, input, acc, bsk) };
        Ok(())
    }

    unsafe fn discard_bootstrap_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
        acc: &GlweCiphertext32,
        bsk: &FourierLweBootstrapKey32,
    ) {
        let buffers =
            self.get_fourier_u32_buffer(bsk.polynomial_size(), bsk.glwe_dimension().to_glwe_size());
        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0, buffers);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextDiscardingBootstrapEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl
    LweCiphertextDiscardingBootstrapEngine<
        FourierLweBootstrapKey64,
        GlweCiphertext64,
        LweCiphertext64,
        LweCiphertext64,
    > for CoreEngine
{
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
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = 3_u64 << 50;
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, lwe_dim_output, glwe_dim, poly_size) = (
    ///     LweDimension(4),
    ///     LweDimension(1024),
    ///     GlweDimension(1),
    ///     PolynomialSize(1024),
    /// );
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// // A constant function is applied during the bootstrap
    /// let lut = vec![8_u64 << 50; poly_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    /// let bsk: FourierLweBootstrapKey64 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// let lwe_sk_output: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dim_output)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let plaintext_vector = engine.create_plaintext_vector(&lut)?;
    /// let acc =
    ///     engine.trivially_encrypt_glwe_ciphertext(glwe_dim.to_glwe_size(), &plaintext_vector)?;
    /// let input = engine.encrypt_lwe_ciphertext(&lwe_sk, &plaintext, noise)?;
    /// let mut output = engine.encrypt_lwe_ciphertext(&lwe_sk_output, &plaintext, noise)?;
    ///
    /// engine.discard_bootstrap_lwe_ciphertext(&mut output, &input, &acc, &bsk)?;
    /// #
    /// assert_eq!(output.lwe_dimension(), lwe_dim_output);
    ///
    /// engine.destroy(lwe_sk)?;
    /// engine.destroy(glwe_sk)?;
    /// engine.destroy(bsk)?;
    /// engine.destroy(lwe_sk_output)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(acc)?;
    /// engine.destroy(input)?;
    /// engine.destroy(output)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
        acc: &GlweCiphertext64,
        bsk: &FourierLweBootstrapKey64,
    ) -> Result<(), LweCiphertextDiscardingBootstrapError<Self::EngineError>> {
        if !ALLOWED_POLY_SIZE.contains(&acc.polynomial_size().0) {
            return Err(LweCiphertextDiscardingBootstrapError::from(
                CoreError::UnsupportedPolynomialSize,
            ));
        }
        LweCiphertextDiscardingBootstrapError::perform_generic_checks(output, input, acc, bsk)?;
        unsafe { self.discard_bootstrap_lwe_ciphertext_unchecked(output, input, acc, bsk) };
        Ok(())
    }

    unsafe fn discard_bootstrap_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
        acc: &GlweCiphertext64,
        bsk: &FourierLweBootstrapKey64,
    ) {
        let buffers =
            self.get_fourier_u64_buffer(bsk.polynomial_size(), bsk.glwe_dimension().to_glwe_size());

        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0, buffers);
    }
}
