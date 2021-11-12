use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweCiphertext32, LweCiphertext64};
use crate::specification::engines::{
    LweCiphertextFusingNegationEngine, LweCiphertextFusingNegationError,
};

impl LweCiphertextFusingNegationEngine<LweCiphertext32> for CoreEngine {
    fn fuse_neg_lwe_ciphertext(
        &mut self,
        input: &mut LweCiphertext32,
    ) -> Result<(), LweCiphertextFusingNegationError<Self::EngineError>> {
        unsafe { self.fuse_neg_lwe_ciphertext_unchecked(input) };
        Ok(())
    }

    unsafe fn fuse_neg_lwe_ciphertext_unchecked(&mut self, input: &mut LweCiphertext32) {
        input.0.update_with_neg();
    }
}

impl LweCiphertextFusingNegationEngine<LweCiphertext64> for CoreEngine {
    fn fuse_neg_lwe_ciphertext(
        &mut self,
        input: &mut LweCiphertext64,
    ) -> Result<(), LweCiphertextFusingNegationError<Self::EngineError>> {
        unsafe { self.fuse_neg_lwe_ciphertext_unchecked(input) };
        Ok(())
    }

    unsafe fn fuse_neg_lwe_ciphertext_unchecked(&mut self, input: &mut LweCiphertext64) {
        input.0.update_with_neg();
    }
}
