use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextVectorEntity, GlweSecretKeyEntity};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::GlweCiphertextCount;

engine_error! {
    GlweCiphertextVectorZeroEncryptionError for GlweCiphertextVectorZeroEncryptionEngine @
    NullCiphertextCount => "The ciphertext count must be greater than zero."
}

/// A trait for engines encrypting zero in GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext vector containing
/// encryptions of zeros, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextVectorZeroEncryptionEngine<SecretKey, CiphertextVector>:
    AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    CiphertextVector: GlweCiphertextVectorEntity<KeyFlavor = SecretKey::KeyFlavor>,
{
    /// Encrypts zero in a GLWE ciphertext vector.
    fn zero_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &SecretKey,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> Result<CiphertextVector, GlweCiphertextVectorZeroEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts zero in a GLWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorZeroEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn zero_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &SecretKey,
        noise: Variance,
        count: GlweCiphertextCount,
    ) -> CiphertextVector;
}
