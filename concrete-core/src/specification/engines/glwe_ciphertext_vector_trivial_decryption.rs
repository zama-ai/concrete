use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextVectorEntity, PlaintextVectorEntity};

engine_error! {
    GlweCiphertextVectorTrivialDecryptionError for GlweCiphertextVectorTrivialDecryptionEngine @
}

/// A trait for engines trivially decrypting GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a plaintext vector containing the
/// trivial decryption of the `input` ciphertext vector.
///
/// # Formal Definition
///
/// see [here](../engines/trait.GlweCiphertextTrivialEncryptionEngine.html)
pub trait GlweCiphertextVectorTrivialDecryptionEngine<CiphertextVector, PlaintextVector>:
    AbstractEngine
where
    CiphertextVector: GlweCiphertextVectorEntity,
    PlaintextVector: PlaintextVectorEntity,
{
    /// Decrypts a GLWE ciphertext vector into a plaintext vector.
    fn trivially_decrypt_glwe_ciphertext_vector(
        &mut self,
        input: &CiphertextVector,
    ) -> Result<PlaintextVector, GlweCiphertextVectorTrivialDecryptionError<Self::EngineError>>;

    /// Unsafely trivially decrypts a GLWE ciphertext vector into a plaintext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorTrivialDecryptionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn trivially_decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        input: &CiphertextVector,
    ) -> PlaintextVector;
}
