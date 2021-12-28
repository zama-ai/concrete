use super::engine_error;
use crate::prelude::AbstractEngine;

use crate::specification::entities::{GlweSecretKeyEntity, LweSecretKeyEntity};

engine_error! {
    GlweToLweSecretKeyTransmutationEngineError for GlweToLweSecretKeyTransmutationEngine @
}

/// A trait for engines transmuting GLWE secret keys into LWE secret keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation moves the existing GLWE into a fresh LWE secret
/// key.
///
/// # Formal Definition
pub trait GlweToLweSecretKeyTransmutationEngine<InputKey, OutputKey>: AbstractEngine
where
    InputKey: GlweSecretKeyEntity,
    OutputKey: LweSecretKeyEntity<KeyDistribution = InputKey::KeyDistribution>,
{
    /// Does the transmutation of the GLWE secret key into a LWE secret key
    fn transmute_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        glwe_secret_key: InputKey,
    ) -> Result<OutputKey, GlweToLweSecretKeyTransmutationEngineError<Self::EngineError>>;

    /// Unsafely transmutes a GLWE secret key into a lwe secret key
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweToLweSecretKeyTransmutationEngineError`].
    /// For safety concerns _specific_ to an engine, refer to the implementer safety section.
    unsafe fn transmute_glwe_secret_key_to_lwe_secret_key_unchecked(
        &mut self,
        glwe_secret_key: InputKey,
    ) -> OutputKey;
}
