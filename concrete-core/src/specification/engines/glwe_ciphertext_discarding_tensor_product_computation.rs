use super::engine_error;
use crate::prelude::GlweCiphertext32;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextDiscardingTensorProductError for GlweCiphertextDiscardingTensorProductEngine @
    PolynomialSizeMismatch => "The polynomial size of the input and output GLWE ciphertexts must be\
     the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertexts must be the same.",
    OutputGlweDimensionMismatch => "The GLWE dimension of the output ciphertext is incorrect"
}

impl<EngineError: std::error::Error> GlweCiphertextDiscardingTensorProductError<EngineError> {
    pub fn perform_generic_checks<InputCiphertext1, InputCiphertext2, OutputCiphertext>(
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
        output: &OutputCiphertext,
    ) -> Result<(), Self>
    where
        InputCiphertext1: GlweCiphertextEntity,
        InputCiphertext2: GlweCiphertextEntity,
        OutputCiphertext: GlweCiphertextEntity,
    {
        if input1.polynomial_size().0 != input2.polynomial_size().0
            || input1.polynomial_size().0 != output.polynomial_size().0
        {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input1.glwe_dimension().0 != input2.glwe_dimension().0 {
            return Err(Self::InputGlweDimensionMismatch);
        }
        if output.glwe_dimension().0
            != 1 / 2 * input1.glwe_dimension().0 * (3 + input1.glwe_dimension().0)
        {
            return Err(Self::OutputGlweDimensionMismatch);
        }
        Ok(())
    }
}

pub trait GlweCiphertextDiscardingTensorProductEngine<
    InputCiphertext1,
    InputCiphertext2,
    OutputCiphertext,
>: AbstractEngine
{
    fn discard_tensor_product_glwe_ciphertext(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        output: &GlweCiphertext32,
    ) -> Result<(), GlweCiphertextDiscardingTensorProductError<Self::EngineError>> {
        unsafe { Ok(self.discard_tensor_product_glwe_ciphertext_unchecked(input1, input2, output)) }
    }

    unsafe fn discard_tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        output: &GlweCiphertext32,
    ) -> ();
}
