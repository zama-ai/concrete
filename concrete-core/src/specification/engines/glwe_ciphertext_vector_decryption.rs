use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweCiphertextVectorEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};

engine_error! {
    GlweCiphertextVectorDecryptionError for GlweCiphertextVectorDecryptionEngine @
    GlweDimensionMismatch => "The key and input ciphertext vector GLWE dimension must be the same.",
    PolynomialSizeMismatch => "The key and input ciphertext vector polynomial size must be the \
                               same."
}

/// A trait for engines decrypting GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a plaintext vector containing
/// the piece-wise decryptions of the `input` GLWE ciphertext vector, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextVectorDecryptionEngine<SecretKey, CiphertextVector, PlaintextVector>:
    AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    CiphertextVector: GlweCiphertextVectorEntity<KeyDistribution = SecretKey::KeyDistribution>,
    PlaintextVector: PlaintextVectorEntity,
{
    /// Decrypts a GLWE ciphertext vector.
    fn decrypt_glwe_ciphertext_vector(
        &mut self,
        key: &SecretKey,
        input: &CiphertextVector,
    ) -> Result<PlaintextVector, GlweCiphertextVectorDecryptionError<Self::EngineError>>;

    /// Unsafely decrypts a GLWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorDecryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn decrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &SecretKey,
        input: &CiphertextVector,
    ) -> PlaintextVector;
}
