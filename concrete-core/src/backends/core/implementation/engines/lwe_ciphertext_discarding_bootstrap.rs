use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierLweBootstrapKey32, FourierLweBootstrapKey64, GlweCiphertext32, GlweCiphertext64,
    LweCiphertext32, LweCiphertext64,
};
use crate::backends::core::private::crypto::bootstrap::Bootstrap;
use crate::specification::engines::{
    LweCiphertextDiscardingBootstrapEngine, LweCiphertextDiscardingBootstrapError,
};
use crate::specification::entities::{
    GlweCiphertextEntity, LweBootstrapKeyEntity, LweCiphertextEntity,
};

impl
    LweCiphertextDiscardingBootstrapEngine<
        FourierLweBootstrapKey32,
        GlweCiphertext32,
        LweCiphertext32,
        LweCiphertext32,
    > for CoreEngine
{
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
        acc: &GlweCiphertext32,
        bsk: &FourierLweBootstrapKey32,
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
        bsk: &FourierLweBootstrapKey32,
    ) {
        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0);
    }
}

impl
    LweCiphertextDiscardingBootstrapEngine<
        FourierLweBootstrapKey64,
        GlweCiphertext64,
        LweCiphertext64,
        LweCiphertext64,
    > for CoreEngine
{
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
        acc: &GlweCiphertext64,
        bsk: &FourierLweBootstrapKey64,
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
        bsk: &FourierLweBootstrapKey64,
    ) {
        bsk.0.bootstrap(&mut output.0, &input.0, &acc.0);
    }
}
