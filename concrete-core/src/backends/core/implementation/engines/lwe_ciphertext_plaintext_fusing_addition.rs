use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    LweCiphertextPlaintextFusingAdditionEngine, LweCiphertextPlaintextFusingAdditionError,
};

impl LweCiphertextPlaintextFusingAdditionEngine<LweCiphertext32, Plaintext32> for CoreEngine {
    fn fuse_add_lwe_ciphertext_plaintext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Plaintext32,
    ) -> Result<(), LweCiphertextPlaintextFusingAdditionError<Self::EngineError>> {
        unsafe { self.fuse_add_lwe_ciphertext_plaintext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_plaintext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Plaintext32,
    ) {
        output.0.get_mut_body().0 += input.0 .0;
    }
}

impl LweCiphertextPlaintextFusingAdditionEngine<LweCiphertext64, Plaintext64> for CoreEngine {
    fn fuse_add_lwe_ciphertext_plaintext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Plaintext64,
    ) -> Result<(), LweCiphertextPlaintextFusingAdditionError<Self::EngineError>> {
        unsafe { self.fuse_add_lwe_ciphertext_plaintext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_plaintext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Plaintext64,
    ) {
        output.0.get_mut_body().0 += input.0 .0;
    }
}
