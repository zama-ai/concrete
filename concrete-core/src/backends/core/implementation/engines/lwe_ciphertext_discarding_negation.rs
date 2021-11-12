use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweCiphertext32, LweCiphertext64};
use crate::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
use crate::specification::engines::{
    LweCiphertextDiscardingNegationEngine, LweCiphertextDiscardingNegationError,
};
use crate::specification::entities::LweCiphertextEntity;

impl LweCiphertextDiscardingNegationEngine<LweCiphertext32, LweCiphertext32> for CoreEngine {
    fn discard_neg_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
    ) -> Result<(), LweCiphertextDiscardingNegationError<Self::EngineError>> {
        if output.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextDiscardingNegationError::LweDimensionMismatch);
        }
        unsafe { self.discard_neg_lwe_ciphertext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn discard_neg_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
    ) {
        output.0.as_mut_tensor().fill_with_copy(input.0.as_tensor());
        output.0.update_with_neg();
    }
}

impl LweCiphertextDiscardingNegationEngine<LweCiphertext64, LweCiphertext64> for CoreEngine {
    fn discard_neg_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
    ) -> Result<(), LweCiphertextDiscardingNegationError<Self::EngineError>> {
        if output.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextDiscardingNegationError::LweDimensionMismatch);
        }
        unsafe { self.discard_neg_lwe_ciphertext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn discard_neg_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
    ) {
        output.0.as_mut_tensor().fill_with_copy(input.0.as_tensor());
        output.0.update_with_neg();
    }
}
