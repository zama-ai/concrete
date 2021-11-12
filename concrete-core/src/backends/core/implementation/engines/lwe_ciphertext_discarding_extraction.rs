use concrete_commons::parameters::MonomialDegree;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, LweCiphertext32, LweCiphertext64,
};
use crate::specification::engines::{
    LweCiphertextDiscardingExtractionEngine, LweCiphertextDiscardingExtractionError,
};
use crate::specification::entities::GlweCiphertextEntity;

impl LweCiphertextDiscardingExtractionEngine<GlweCiphertext32, LweCiphertext32> for CoreEngine {
    fn discard_extract_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &GlweCiphertext32,
        nth: MonomialDegree,
    ) -> Result<(), LweCiphertextDiscardingExtractionError<Self::EngineError>> {
        if output.0.lwe_size().to_lwe_dimension().0
            != input.0.polynomial_size().0 * input.0.size().to_glwe_dimension().0
        {
            return Err(LweCiphertextDiscardingExtractionError::SizeMismatch);
        }
        if nth.0 > input.glwe_dimension().0 - 1 {
            return Err(LweCiphertextDiscardingExtractionError::MonomialDegreeTooLarge);
        }
        unsafe { self.discard_extract_lwe_ciphertext_unchecked(output, input, nth) };
        Ok(())
    }

    unsafe fn discard_extract_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &GlweCiphertext32,
        nth: MonomialDegree,
    ) {
        output.0.fill_with_glwe_sample_extraction(&input.0, nth);
    }
}

impl LweCiphertextDiscardingExtractionEngine<GlweCiphertext64, LweCiphertext64> for CoreEngine {
    fn discard_extract_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &GlweCiphertext64,
        nth: MonomialDegree,
    ) -> Result<(), LweCiphertextDiscardingExtractionError<Self::EngineError>> {
        if output.0.lwe_size().to_lwe_dimension().0
            != input.0.polynomial_size().0 * input.0.size().to_glwe_dimension().0
        {
            return Err(LweCiphertextDiscardingExtractionError::SizeMismatch);
        }
        if nth.0 > input.glwe_dimension().0 - 1 {
            return Err(LweCiphertextDiscardingExtractionError::MonomialDegreeTooLarge);
        }
        unsafe { self.discard_extract_lwe_ciphertext_unchecked(output, input, nth) };
        Ok(())
    }

    unsafe fn discard_extract_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &GlweCiphertext64,
        nth: MonomialDegree,
    ) {
        output.0.fill_with_glwe_sample_extraction(&input.0, nth);
    }
}
