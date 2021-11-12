use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    Cleartext32, Cleartext64, LweCiphertext32, LweCiphertext64,
};
use crate::specification::engines::{
    LweCiphertextCleartextFusingMultiplicationEngine,
    LweCiphertextCleartextFusingMultiplicationError,
};

impl LweCiphertextCleartextFusingMultiplicationEngine<LweCiphertext32, Cleartext32> for CoreEngine {
    fn fuse_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Cleartext32,
    ) -> Result<(), LweCiphertextCleartextFusingMultiplicationError<Self::EngineError>> {
        unsafe { self.fuse_mul_lwe_ciphertext_cleartext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Cleartext32,
    ) {
        output.0.update_with_scalar_mul(input.0);
    }
}

impl LweCiphertextCleartextFusingMultiplicationEngine<LweCiphertext64, Cleartext64> for CoreEngine {
    fn fuse_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Cleartext64,
    ) -> Result<(), LweCiphertextCleartextFusingMultiplicationError<Self::EngineError>> {
        unsafe { self.fuse_mul_lwe_ciphertext_cleartext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Cleartext64,
    ) {
        output.0.update_with_scalar_mul(input.0);
    }
}
