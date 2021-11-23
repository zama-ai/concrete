use super::engine_error;

use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, LweSecretKeyEntity, PlaintextEntity};

engine_error! {
    LweCiphertextDecryptionError for LweCiphertextDecryptionEngine @
}

/// A trait for engines decrypting LWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates an plaintext containing the
/// decryption of the `input` LWE ciphertext, under the `key` secret key.
///
/// # Formal Definition
pub trait LweCiphertextDecryptionEngine<SecretKey, Ciphertext, Plaintext>: AbstractEngine
where
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
    Plaintext: PlaintextEntity,
{
    /// Decrypts an LWE ciphertext.
    fn decrypt_lwe_ciphertext(
        &mut self,
        key: &SecretKey,
        input: &Ciphertext,
    ) -> Result<Plaintext, LweCiphertextDecryptionError<Self::EngineError>>;

    /// Unsafely decrypts an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDecryptionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn decrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        input: &Ciphertext,
    ) -> Plaintext;
}
