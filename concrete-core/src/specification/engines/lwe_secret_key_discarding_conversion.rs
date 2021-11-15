use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweSecretKeyEntity;

engine_error! {
    LweSecretKeyDiscardingConversionError for LweSecretKeyDiscardingConversionEngine @
    LweDimensionMismatch => "The input and output LWE dimensions must be the same."
}

/// A trait for engines converting (discarding) LWE secret keys .
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE secret key with
/// the conversion of the `input` LWE secret key to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait LweSecretKeyDiscardingConversionEngine<Input, Output>: AbstractEngine
where
    Input: LweSecretKeyEntity,
    Output: LweSecretKeyEntity<KeyFlavor = Input::KeyFlavor>,
{
    /// Converts a LWE secret key .
    fn discard_convert_lwe_secret_key(
        &mut self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), LweSecretKeyDiscardingConversionError<Self::EngineError>>;

    /// Unsafely converts a LWE secret key .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweSecretKeyDiscardingConversionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_convert_lwe_secret_key_unchecked(
        &mut self,
        output: &mut Output,
        input: &Input,
    );
}
