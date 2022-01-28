use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, GlweRelinearizationKeyEntity};

engine_error! {
    GlweCiphertextDiscardingRelinearizationError for GlweCiphertextDiscardingRelinearizationEngine@
    PolynomialSizeMismatch => "The polynomial size of the input and output GLWE ciphertexts must \
    be the same.",
    InputGlweDimensionMismatch => "The GLWE dimension of the input ciphertext and key must be the \
    same.",
    OutputGlweDimensionMismatch => "The GLWE dimension of the output ciphertext is incorrect"
}

impl<EngineError: std::error::Error> GlweCiphertextDiscardingRelinearizationError<EngineError> {
    pub fn perform_generic_checks<InputKey, InputCiphertext, OutputCiphertext>(
        input_key: &InputKey,
        input_ct: &InputCiphertext,
        output: &OutputCiphertext,
    ) -> Result<(), Self>
    where
        InputKey: GlweRelinearizationKeyEntity,
        InputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
        OutputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    {
        if input_ct.polynomial_size().0 != input_key.polynomial_size().0
            || input_ct.polynomial_size().0 != output.polynomial_size().0
        {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input_ct.glwe_dimension().0 != input_key.glwe_dimension().0 {
            return Err(Self::InputGlweDimensionMismatch);
        }
        if 2 * input_ct.glwe_dimension().0
            != output.glwe_dimension().0 * (3 + output.glwe_dimension().0)
        {
            return Err(Self::OutputGlweDimensionMismatch);
        }
        Ok(())
    }
}

/// A trait for engines performing a (discarding) relinearization on a GLWE ciphertext.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext with
/// the relinearization of the `input` GLWE ciphertext, using the `input` relinearization key
///
/// # Formal Definition
pub trait GlweCiphertextDiscardingRelinearizationEngine<InputKey, InputCiphertext, OutputCiphertext>:
    AbstractEngine
where
    InputKey: GlweRelinearizationKeyEntity,
    InputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
    OutputCiphertext: GlweCiphertextEntity<KeyDistribution = InputKey::KeyDistribution>,
{
    fn discard_relinearize_glwe_ciphertext(
        &mut self,
        input_key: &InputKey,
        input_ct: &InputCiphertext,
        output: &mut OutputCiphertext,
    ) -> Result<(), GlweCiphertextDiscardingRelinearizationError<Self::EngineError>>;

    /// Unsafely performs a discarding reliniearization of two GLWE ciphertexts.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextDiscardingRelinearizationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_relinearize_glwe_ciphertext_unchecked(
        &mut self,
        input_key: &InputKey,
        input_ct: &InputCiphertext,
        output: &mut OutputCiphertext,
    );
}
