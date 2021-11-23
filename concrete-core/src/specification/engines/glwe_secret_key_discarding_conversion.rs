use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweSecretKeyEntity;

engine_error! {
    GlweSecretKeyDiscardingConversionError for GlweSecretKeyDiscardingConversionEngine @
    GlweDimensionMismatch => "The input and output GLWE dimension must be the same.",
    PolynomialSizeMismatch => "The input and output polynomial size must be the same."
}

/// A trait for engines converting (discarding) GLWE secret keys .
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE secret key with
/// the conversion of the `input` GLWE secret key to a type with a different representation (for
/// instance from cpu to gpu memory).
///
/// # Formal Definition
pub trait GlweSecretKeyDiscardingConversionEngine<Input, Output>: AbstractEngine
where
    Input: GlweSecretKeyEntity,
    Output: GlweSecretKeyEntity<KeyDistribution = Input::KeyDistribution>,
{
    /// Converts a GLWE secret key .
    fn discard_convert_glwe_secret_key(
        &mut self,
        output: &mut Output,
        input: &Input,
    ) -> Result<(), GlweSecretKeyDiscardingConversionError<Self::EngineError>>;

    /// Unsafely converts a GLWE secret key .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweSecretKeyDiscardingConversionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_convert_glwe_secret_key_unchecked(
        &mut self,
        output: &mut Output,
        input: &Input,
    );
}
