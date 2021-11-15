use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorDiscardingConversionError for LweCiphertextVectorDiscardingConversionEngine @
    LweDimensionMismatch => "The input and output LWE dimension must be the same.",
    CiphertextCountMismatch => "The input and output ciphertext count must be the same."
}

/// A trait for engines converting (discarding) LWE ciphertext vectors .
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext vector
/// with the conversion of the `input` LWE ciphertext vector to a type with a different
/// representation (for instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait LweCiphertextVectorDiscardingConversionEngine<Input, Output>: AbstractEngine
where
    Input: LweCiphertextVectorEntity,
    Output: LweCiphertextVectorEntity<KeyFlavor = Input::KeyFlavor>,
{
    /// Converts a LWE ciphertext vector .
    fn discard_convert_lwe_ciphertext_vector(
        &mut self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), LweCiphertextVectorDiscardingConversionError<Self::EngineError>>;

    /// Unsafely converts a LWE ciphertext vector .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorDiscardingConversionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_convert_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut Output,
        input: &Input,
    );
}
