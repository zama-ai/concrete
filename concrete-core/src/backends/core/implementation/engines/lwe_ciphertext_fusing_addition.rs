use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweCiphertext32, LweCiphertext64};
use crate::specification::engines::{
    LweCiphertextFusingAdditionEngine, LweCiphertextFusingAdditionError,
};
use crate::specification::entities::LweCiphertextEntity;

impl LweCiphertextFusingAdditionEngine<LweCiphertext32, LweCiphertext32> for CoreEngine {
    fn fuse_add_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
    ) -> Result<(), LweCiphertextFusingAdditionError<Self::EngineError>> {
        if output.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextFusingAdditionError::LweDimensionMismatch);
        }
        unsafe { self.fuse_add_lwe_ciphertext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &LweCiphertext32,
    ) {
        output.0.update_with_add(&input.0);
    }
}

impl LweCiphertextFusingAdditionEngine<LweCiphertext64, LweCiphertext64> for CoreEngine {
    fn fuse_add_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
    ) -> Result<(), LweCiphertextFusingAdditionError<Self::EngineError>> {
        if output.lwe_dimension() != input.lwe_dimension() {
            return Err(LweCiphertextFusingAdditionError::LweDimensionMismatch);
        }
        unsafe { self.fuse_add_lwe_ciphertext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &LweCiphertext64,
    ) {
        output.0.update_with_add(&input.0);
    }
}
