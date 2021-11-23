use super::engine_error;

use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, LweSecretKeyEntity, PlaintextEntity};
use concrete_commons::dispersion::Variance;

engine_error! {
    LweCiphertextDiscardingEncryptionError for LweCiphertextDiscardingEncryptionEngine @
    LweDimensionMismatch => "The secret key and ciphertext LWE dimensions must be the same."
}

/// A trait for engines encrypting (discarding) LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// encryption of the `input` plaintext, under the `key` secret key.
///
/// # Formal Definition
pub trait LweCiphertextDiscardingEncryptionEngine<SecretKey, Plaintext, Ciphertext>:
    AbstractEngine
where
    SecretKey: LweSecretKeyEntity,
    Plaintext: PlaintextEntity,
    Ciphertext: LweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    /// Encrypts an LWE ciphertext.
    fn discard_encrypt_lwe_ciphertext(
        &mut self,
        key: &SecretKey,
        output: &mut Ciphertext,
        input: &Plaintext,
        noise: Variance,
    ) -> Result<(), LweCiphertextDiscardingEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        output: &mut Ciphertext,
        input: &Plaintext,
        noise: Variance,
    );
}
