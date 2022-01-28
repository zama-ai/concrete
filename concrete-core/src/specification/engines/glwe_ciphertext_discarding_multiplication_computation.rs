use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, GlweRelinearizationKeyEntity};

engine_error! {
    GlweCiphertextDiscardingMultiplicationError for GlweCiphertextDiscardingMultiplicationEngine @
    PolynomialSizeMismatch => "The polynomial size of the input key, input and output GLWE\
     ciphertexts must be the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertexts and key must be the \
    same.",
    OutputGlweDimensionMismatch => "The GLWE dimension of the output ciphertext is incorrect"
}

impl<EngineError: std::error::Error> GlweCiphertextDiscardingMultiplicationError<EngineError> {
    pub fn perform_generic_checks<InputKey, InputCiphertext1, InputCiphertext2, OutputCiphertext>(
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
        output: &OutputCiphertext,
    ) -> Result<(), Self>
    where
        InputKey: GlweRelinearizationKeyEntity,
        InputCiphertext1: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
        InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
        OutputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    {
        if input1.polynomial_size().0 != input2.polynomial_size().0
            || input1.polynomial_size().0 != output.polynomial_size().0
            || input_key.polynomial_size().0 != input1.polynomial_size().0
        {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input1.glwe_dimension().0 != input2.glwe_dimension().0
            || input_key.glwe_dimension().0 != input1.glwe_dimension().0
        {
            return Err(Self::InputGlweDimensionMismatch);
        }
        if output.glwe_dimension().0 != input1.glwe_dimension().0
            || input_key.glwe_dimension().0 != output.glwe_dimension().0
        {
            return Err(Self::OutputGlweDimensionMismatch);
        }
        Ok(())
    }
}

/// A trait for engines multiplying (discarding) GLWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext with
/// the multiplication of the `input` GLWE ciphertexts, using the `input` relinearisation key.
///
/// # Formal Definition
pub trait GlweCiphertextDiscardingMultiplicationEngine<
    InputKey,
    InputCiphertext1,
    InputCiphertext2,
    OutputCiphertext,
>: AbstractEngine where
    InputKey: GlweRelinearizationKeyEntity,
    InputCiphertext1: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    OutputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
{
    fn discard_multiply_glwe_ciphertext(
        &mut self,
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
        output: &mut OutputCiphertext,
    ) -> Result<(), GlweCiphertextDiscardingMultiplicationError<Self::EngineError>>;

    /// Unsafely performs a discarding multiplication of two GLWE ciphertexts.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextDiscardingMultiplicationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.

    unsafe fn discard_multiply_glwe_ciphertext_unchecked(
        &mut self,
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
        output: &mut OutputCiphertext,
    );
}
