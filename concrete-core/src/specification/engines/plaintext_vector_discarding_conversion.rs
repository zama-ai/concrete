use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::PlaintextVectorEntity;

engine_error! {
    PlaintextVectorDiscardingConversionError for PlaintextVectorDiscardingConversionEngine @
    PlaintextCountMismatch => "The input and output plaintext count must be the same"
}

/// A trait for engines converting (discarding) plaintext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` plaintext vector with
/// the conversion of the `input` plaintext vector to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait PlaintextVectorDiscardingConversionEngine<Input, Output>: AbstractEngine
where
    Input: PlaintextVectorEntity,
    Output: PlaintextVectorEntity,
{
    /// Converts a plaintext vector .
    fn discard_convert_plaintext_vector(
        &mut self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), PlaintextVectorDiscardingConversionError<Self::EngineError>>;

    /// Unsafely converts a plaintext vector .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`PlaintextVectorDiscardingConversionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_convert_plaintext_vector_unchecked(
        &mut self,
        output: &mut Output,
        input: &Input,
    );
}
