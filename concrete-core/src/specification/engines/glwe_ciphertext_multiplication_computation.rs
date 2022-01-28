use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextMultiplicationError for GlweCiphertextMultiplicationEngine @
    PolynomialSizeMismatch => "The polynomial size of the input GLWE ciphertexts must be the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertexts must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextMultiplicationError<EngineError> {
    pub fn perform_generic_checks<InputCiphertext1, InputCiphertext2>(
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    ) -> Result<(), Self>
    where
        InputCiphertext1: GlweCiphertextEntity,
        InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputCiphertext1::KeyDistribution>,
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
/// A trait for engines multiplying GLWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext with
/// the multiplication of the `input` GLWE ciphertexts, using the `input` relinearisation key.
///
/// # Formal Definition
pub trait GlweCiphertextMultiplicationEngine<InputCiphertext1, InputCiphertext2>:
    AbstractEngine
where
    InputCiphertext1: GlweCiphertextEntity,
    InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputCiphertext1::KeyDistribution>,
{
    fn multiply_glwe_ciphertext(
        &mut self,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    ) -> Result<(), GlweCiphertextMultiplicationError<Self::EngineError>>;

    /// Unsafely performs a multiplication of two GLWE ciphertexts.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextMultiplicationError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.

    unsafe fn multiply_glwe_ciphertext_unchecked(
        &mut self,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    );
}
