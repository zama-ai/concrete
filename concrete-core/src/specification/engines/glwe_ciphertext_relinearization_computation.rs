use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, GlweRelinearizationKeyEntity};

engine_error! {
    GlweCiphertextRelinearizationError for GlweCiphertextRelinearizationEngine@
    PolynomialSizeMismatch => "The polynomial size of the input and output GLWE ciphertexts must \
    be the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertexts must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextRelinearizationError<EngineError> {
    pub fn perform_generic_checks<InputKey, InputCiphertext1, InputCiphertext2>(
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    ) -> Result<(), Self>
    where
        InputKey: GlweRelinearizationKeyEntity,
        InputCiphertext1: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
        InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    {
        if input1.polynomial_size().0 != input2.polynomial_size().0
            || input1.polynomial_size().0 != input_key.polynomial_size().0
        {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input1.glwe_dimension().0 != input2.glwe_dimension().0
            || input1.glwe_dimension().0 != input_key.glwe_dimension().0
        {
            return Err(Self::InputGlweDimensionMismatch);
        }
        Ok(())
    }
}
/// A trait for engines performing a relinearization on a GLWE ciphertext.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) generates a GLWE ciphertext with
/// the relinearization of the `input` GLWE ciphertexts, using the `input` relinearization key
///
/// # Formal Definition
pub trait GlweCiphertextRelinearizationEngine<InputKey, InputCiphertext1, InputCiphertext2>:
    AbstractEngine
where
    InputKey: GlweRelinearizationKeyEntity,
    InputCiphertext1: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    InputCiphertext2: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    ) -> Result<(), GlweCiphertextRelinearizationError<Self::EngineError>>;

    /// Unsafely performs a relinearization of a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextRelinearizationError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.

    unsafe fn multiply_glwe_ciphertext_unchecked(
        &mut self,
        input_key: &InputKey,
        input1: &InputCiphertext1,
        input2: &InputCiphertext2,
    );
}
