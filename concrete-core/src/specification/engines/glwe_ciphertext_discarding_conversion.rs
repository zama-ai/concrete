use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextDiscardingConversionError for GlweCiphertextDiscardingConversionEngine @
    GlweDimensionMismatch => "The input and output GLWE dimension must be the same.",
    PolynomialSizeMismatch => "The input and output polynomial size must be the same."
}

/// A trait for engines converting (discarding) GLWE ciphertexts .
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext with
/// the conversion of the `input` GLWE ciphertext to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait GlweCiphertextDiscardingConversionEngine<Input, Output>: AbstractEngine
where
    Input: GlweCiphertextEntity,
    Output: GlweCiphertextEntity<KeyFlavor = Input::KeyFlavor>,
{
    /// Converts a GLWE ciphertext .
    fn discard_convert_glwe_ciphertext(
        &mut self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), GlweCiphertextDiscardingConversionError<Self::EngineError>>;

    /// Unsafely converts a GLWE ciphertext .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextDiscardingConversionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_convert_glwe_ciphertext_unchecked(
        &mut self,
        output: &mut Output,
        input: &Input,
    );
}
