use super::engine_error;
use crate::prelude::GlweCiphertext32;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextTensorProductError for GlweCiphertextTensorProductEngine @
    PolynomialSizeMismatch => "The polynomial size of the input and output GLWE ciphertexts must be\
     the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertexts must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextTensorProductError<EngineError> {
    pub fn perform_generic_checks<InputCiphertext1, InputCiphertext2>(
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    ) -> Result<(), Self>
    where
        InputCiphertext1: GlweCiphertextEntity,
        InputCiphertext2: GlweCiphertextEntity,
    {
        if input1.polynomial_size().0 != input2.polynomial_size().0 {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input1.glwe_dimension().0 != input2.glwe_dimension().0 {
            return Err(Self::InputGlweDimensionMismatch);
        }
        Ok(())
    }
}

pub trait GlweCiphertextTensorProductEngine<InputCiphertext1, InputCiphertext2>:
    AbstractEngine
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
    ) -> Result<(), GlweCiphertextTensorProductError<Self::EngineError>> {
        unsafe { Ok(self.tensor_product_glwe_ciphertext_unchecked(input1, input2)) }
    }

    unsafe fn tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
    ) -> ();
}
