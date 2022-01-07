use crate::backends::core::entities::{
    GlweCiphertext32, GlweCiphertext64, LweCiphertext32, LweCiphertext64,
};
use crate::backends::optalysys::implementation::engines::OptalysysEngine;
use crate::backends::optalysys::implementation::entities::{
    OptalysysFourierLweBootstrapKey32, OptalysysFourierLweBootstrapKey64,
};
use crate::specification::engines::{
    LweCiphertextDiscardingBootstrapEngine, LweCiphertextDiscardingBootstrapError,
};
use crate::specification::entities::{
    GlweCiphertextEntity, LweBootstrapKeyEntity, LweCiphertextEntity,
};

/// # Description:
/// Implementation of [`LweCiphertextDiscardingBootstrapEngine`] for [`OptalysysEngine`] that
/// operates on 32 bits integers.
impl
    LweCiphertextDiscardingBootstrapEngine<
        OptalysysFourierLweBootstrapKey32,
        GlweCiphertext32,
        LweCiphertext32,
        LweCiphertext32,
    > for OptalysysEngine
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
    /// let acc = engine.encrypt_glwe_ciphertext(&glwe_sk, &plaintext_vector, noise)?;
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
        bsk: &OptalysysFourierLweBootstrapKey32,
    ) -> Result<(), LweCiphertextDiscardingBootstrapError<Self::EngineError>> {
        if input.lwe_dimension() != bsk.input_lwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::InputLweDimensionMismatch);
        }
        if acc.polynomial_size() != bsk.polynomial_size() {
            return Err(LweCiphertextDiscardingBootstrapError::AccumulatorPolynomialSizeMismatch);
        }
        if acc.glwe_dimension() != bsk.glwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::AccumulatorGlweDimensionMismatch);
        }
        if output.lwe_dimension() != bsk.output_lwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::OutputLweDimensionMismatch);
        }
        unsafe { self.discard_bootstrap_lwe_ciphertext_unchecked(output, input, acc, bsk) };
        Ok(())
    }

    unsafe fn discard_bootstrap_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
        acc: &GlweCiphertext32,
        bsk: &OptalysysFourierLweBootstrapKey32,
    ) {
        let buffers = self.get_fourier_bootstrap_u32_buffer(bsk);
        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0, buffers);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextDiscardingBootstrapEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl
    LweCiphertextDiscardingBootstrapEngine<
        OptalysysFourierLweBootstrapKey64,
        GlweCiphertext64,
        LweCiphertext64,
        LweCiphertext64,
    > for OptalysysEngine
{
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
        acc: &GlweCiphertext64,
        bsk: &OptalysysFourierLweBootstrapKey64,
    ) -> Result<(), LweCiphertextDiscardingBootstrapError<Self::EngineError>> {
        if input.lwe_dimension() != bsk.input_lwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::InputLweDimensionMismatch);
        }
        if acc.polynomial_size() != bsk.polynomial_size() {
            return Err(LweCiphertextDiscardingBootstrapError::AccumulatorPolynomialSizeMismatch);
        }
        if acc.glwe_dimension() != bsk.glwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::AccumulatorGlweDimensionMismatch);
        }
        if output.lwe_dimension() != bsk.output_lwe_dimension() {
            return Err(LweCiphertextDiscardingBootstrapError::OutputLweDimensionMismatch);
        }
        unsafe { self.discard_bootstrap_lwe_ciphertext_unchecked(output, input, acc, bsk) };
        Ok(())
    }

    unsafe fn discard_bootstrap_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
        acc: &GlweCiphertext64,
        bsk: &OptalysysFourierLweBootstrapKey64,
    ) {
        let buffers = self.get_fourier_bootstrap_u64_buffer(bsk);

        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0, buffers);
    }
}
