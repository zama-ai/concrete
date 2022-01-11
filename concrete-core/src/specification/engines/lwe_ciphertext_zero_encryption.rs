use super::engine_error;

use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, LweSecretKeyEntity};
use concrete_commons::dispersion::Variance;

engine_error! {
    LweCiphertextZeroEncryptionError for LweCiphertextZeroEncryptionEngine @
}

/// A trait for engines encrypting zero in LWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates an LWE ciphertext containing an
/// encryption of zero, under the `key` secret key.
///
/// # Formal Definition
pub trait LweCiphertextZeroEncryptionEngine<SecretKey, Ciphertext>: AbstractEngine
where
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    /// Encrypts zero into an LWE ciphertext.
    fn zero_encrypt_lwe_ciphertext(
        &mut self,
        key: &SecretKey,
        noise: Variance,
    ) -> Result<Ciphertext, LweCiphertextZeroEncryptionError<Self::EngineError>>;

    /// Safely encrypts zero into an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextZeroEncryptionError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.
    unsafe fn zero_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        noise: Variance,
    ) -> Ciphertext;
}
