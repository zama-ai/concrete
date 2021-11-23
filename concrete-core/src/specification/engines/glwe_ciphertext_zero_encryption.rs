use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, GlweSecretKeyEntity};
use concrete_commons::dispersion::Variance;

engine_error! {
    GlweCiphertextZeroEncryptionError for GlweCiphertextZeroEncryptionEngine @
}

/// A trait for engines encrypting zeros in GLWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext containing an
/// encryption of zeros, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextZeroEncryptionEngine<SecretKey, Ciphertext>: AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    Ciphertext: GlweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    /// Encrypts a zero in a GLWE ciphertext.
    fn zero_encrypt_glwe_ciphertext(
        &mut self,
        key: &SecretKey,
        noise: Variance,
    ) -> Result<Ciphertext, GlweCiphertextZeroEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts a zero in a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextZeroEncryptionError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.
    unsafe fn zero_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        noise: Variance,
    ) -> Ciphertext;
}
