use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweCiphertextVectorEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};
use concrete_commons::dispersion::Variance;

engine_error! {
    GlweCiphertextVectorEncryptionError for GlweCiphertextVectorEncryptionEngine @
    PlaintextCountMismatch => "The key polynomial size must divide the plaintext count of the input \
                               vector."
}

/// A trait for engines encrypting GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext vector containing
/// the piece-wise encryptions of the `input` plaintext vector, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextVectorEncryptionEngine<SecretKey, PlaintextVector, CiphertextVector>:
    AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: GlweCiphertextVectorEntity<KeyFlavor = SecretKey::KeyFlavor>,
{
    /// Encrypts a GLWE ciphertext vector.
    fn encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &SecretKey,
        input: &PlaintextVector,
        noise: Variance,
    ) -> Result<CiphertextVector, GlweCiphertextVectorEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts a GLWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &SecretKey,
        input: &PlaintextVector,
        noise: Variance,
    ) -> CiphertextVector;
}
