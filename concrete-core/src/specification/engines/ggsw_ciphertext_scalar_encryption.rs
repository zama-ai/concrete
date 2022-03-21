use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GgswCiphertextEntity, GlweSecretKeyEntity, PlaintextEntity};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    GgswCiphertextScalarEncryptionError for GgswCiphertextScalarEncryptionEngine @
}

/// A trait for engines encrypting GGSW ciphertexts containing a single plaintext.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GGSW ciphertext containing the
/// encryption of the `input` plaintext, under the `key` secret key.
///
/// # Formal Definition
pub trait GgswCiphertextScalarEncryptionEngine<SecretKey, Plaintext, Ciphertext>:
    AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
    Plaintext: PlaintextEntity,
    Ciphertext: GgswCiphertextEntity<KeyDistribution = SecretKey::KeyDistribution>,
{
    /// Encrypts a plaintext into a GGSW ciphertext.
    fn encrypt_scalar_ggsw_ciphertext(
        &mut self,
        key: &SecretKey,
        input: &Plaintext,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> Result<Ciphertext, GgswCiphertextScalarEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts a plaintext vector into a GGSW ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GgswCiphertextScalarEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn encrypt_scalar_ggsw_ciphertext_unchecked(
        &mut self,
        key: &SecretKey,
        input: &Plaintext,
        noise: Variance,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
    ) -> Ciphertext;
}
