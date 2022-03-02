use super::engine_error;
use crate::prelude::PlaintextVectorEntity;
use concrete_commons::parameters::GlweSize;

use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextVectorEntity;

engine_error! {
    GlweCiphertextVectorTrivialEncryptionError for GlweCiphertextVectorTrivialEncryptionEngine @
}

/// A trait for engines trivially encrypting GLWE ciphertext vector.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext vector containing the
/// trivial encryption of the `input` plaintext vector with the requested `glwe_size`.
///
/// # Formal Definition
///
/// A trivial encryption uses a zero mask and no noise.
/// It is absolutely not secure, as the body contains a direct copy of the plaintext
pub trait GlweCiphertextVectorTrivialEncryptionEngine<PlaintextVector, CiphertextVector>:
    AbstractEngine
where
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: GlweCiphertextVectorEntity,
{
    /// Trivially encrypts a plaintext vector into a GLWE ciphertext vector.
    fn trivially_encrypt_glwe_ciphertext_vector(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector,
    ) -> Result<CiphertextVector, GlweCiphertextVectorTrivialEncryptionError<Self::EngineError>>;

    /// Unsafely trivially encrypts a plaintext vector into a GLWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorTrivialEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn trivially_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector,
    ) -> CiphertextVector;
}
