use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweRelinearizationKeyEntity, GlweSecretKeyEntity};

engine_error! {
    GlweSecretKeyRelinearizationConversionError for GlweSecretKeyRelinearizationConversionEngine @
}

/// A trait for engines converting a pair of GLWE secret keys into a GLWE relinearization key
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE relinearization key
/// containing the conversion of the two input GLWE secret keys `input1` and `input2`.
///
/// # Formal Definition
///
/// The goal of this function is to take as input two GLWE secret keys s1 and s2, and
/// create a relinearization key {CT_{i,j}}_{i <= j <=i, 1 <= i <=k} of the form
///
/// CT_{i,j} = GLev^{B,l}_s(S_i * S_j)

pub trait GlweSecretKeyRelinearizationConversionEngine<InputKey, OutputKey>:
    AbstractEngine
where
    InputKey: GlweSecretKeyEntity,
    OutputKey: GlweRelinearizationKeyEntity<KeyDistribution = InputKey::KeyDistribution>,
{
    /// Converts a pair of GLWE secret keys to a GLWE relinearisation key
    fn convert_glwe_secret_key_to_relinearization_key(
        &mut self,
        input_key: &InputKey,
    ) -> Result<OutputKey, GlweSecretKeyRelinearizationConversionError<Self::EngineError>>;

    /// Unsafely converts a pair of two GLWE secret keys to a GLWE relinearisation key
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweSecretKeyRelinearizationConversionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn convert_glwe_secret_key_to_relinearization_key_unchecked(
        &mut self,
        input1: &InputKey,
    ) -> OutputKey;
}
