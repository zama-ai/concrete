use crate::backends::core::entities::{
    GlweCiphertext32, GlweCiphertext64, LweCiphertext32, LweCiphertext64,
};
use crate::backends::optalysys::entities::{
    OptalysysFourierLweBootstrapKey32, OptalysysFourierLweBootstrapKey64,
};
use crate::backends::optalysys::implementation::engines::OptalysysEngine;
use crate::prelude::{FourierLweBootstrapKey32, FourierLweBootstrapKey64};
use crate::specification::engines::{
    LweCiphertextDiscardingBootstrapEngine, LweCiphertextDiscardingBootstrapError,
};
use crate::specification::entities::{LweBootstrapKeyEntity, LweCiphertextEntity};

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
    /// let mut engine = OptalysysEngine::new()?;
    /// let mut core_engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey32 = core_engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey32 = core_engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    /// let bsk: LweBootstrapKey32 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// let lwe_sk_output: LweSecretKey32 = core_engine.create_lwe_secret_key(lwe_dim_output)?;
    /// let plaintext = core_engine.create_plaintext(&input)?;
    /// let plaintext_vector = core_engine.create_plaintext_vector(&lut)?;
    /// let acc = core_engine.encrypt_glwe_ciphertext(&glwe_sk, &plaintext_vector, noise)?;
    /// let input = core_engine.encrypt_lwe_ciphertext(&lwe_sk, &plaintext, noise)?;
    /// let mut output = core_engine.zero_encrypt_lwe_ciphertext(&lwe_sk_output, noise)?;
    ///
    /// let optalysys_bsk = engine.convert_lwe_bootstrap_key(&bsk)?;
    /// engine.discard_bootstrap_lwe_ciphertext(&mut output, &input, &acc, &optalysys_bsk)?;
    /// #
    /// assert_eq!(output.lwe_dimension(), lwe_dim_output);
    ///
    /// core_engine.destroy(lwe_sk)?;
    /// core_engine.destroy(glwe_sk)?;
    /// core_engine.destroy(bsk)?;
    /// core_engine.destroy(lwe_sk_output)?;
    /// core_engine.destroy(plaintext)?;
    /// core_engine.destroy(plaintext_vector)?;
    /// core_engine.destroy(acc)?;
    /// core_engine.destroy(input)?;
    /// core_engine.destroy(output)?;
    /// engine.destroy(optalysys_bsk)?;
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
        LweCiphertextDiscardingBootstrapError::perform_generic_checks(output, input, acc, bsk)?;
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
        LweCiphertextDiscardingBootstrapError::perform_generic_checks(output, input, acc, bsk)?;
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
