use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweCiphertextEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};
use concrete_commons::dispersion::Variance;

engine_error! {
    GlweCiphertextEncryptionError for GlweCiphertextEncryptionEngine @
    PlaintextCountMismatch => "The plaintext count of the input vector and the key polynomial size \
                               must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextEncryptionError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks<SecretKey, PlaintextVector>(
        key: &SecretKey,
        input: &PlaintextVector,
    ) -> Result<(), Self>
    where
        SecretKey: GlweSecretKeyEntity,
        PlaintextVector: PlaintextVectorEntity,
    {
        if key.polynomial_size().0 != input.plaintext_count().0 {
            return Err(Self::PlaintextCountMismatch);
        }
        Ok(())
    }
}

/// A trait for engines encrypting GLWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext containing the
/// encryptions of the `input` plaintext vector, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextEncryptionEngine<SecretKey, PlaintextVector, Ciphertext>:
    AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    PlaintextVector: PlaintextVectorEntity,
    Ciphertext: GlweCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    /// Encrypts a plaintext vector into a GLWE ciphertext.
    fn encrypt_glwe_ciphertext(
        &mut self,
        key: &SecretKey,
        input: &PlaintextVector,
        noise: Variance,
    ) -> Result<Ciphertext, GlweCiphertextEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts a plaintext vector into a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextEncryptionError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.
    unsafe fn encrypt_glwe_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        input: &PlaintextVector,
        noise: Variance,
    ) -> Ciphertext;
}
