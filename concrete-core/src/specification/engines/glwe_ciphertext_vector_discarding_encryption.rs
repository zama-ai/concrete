use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweCiphertextVectorEntity, GlweSecretKeyEntity, PlaintextVectorEntity,
};
use concrete_commons::dispersion::Variance;

engine_error! {
    GlweCiphertextVectorDiscardingEncryptionError for GlweCiphertextVectorDiscardingEncryptionEngine @
    GlweDimensionMismatch => "The GLWE dimensions of the key and the output ciphertext vector must \
                              be the same.",
    PolynomialSizeMismatch => "The polynomial size of the key and the output ciphertext vector \
                               must be the same.",
    PlaintextCountMismatch => "The input plaintext vector length and output ciphertext vector \
                               capacity (poly size * length) must be the same."
}

/// A trait for engines encrypting (discarding) GLWE ciphertext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` GLWE ciphertext vector
/// with the piece-wise encryptions of the `input` plaintext vector, under the `key` secret key.
///
/// # Formal Definition
pub trait GlweCiphertextVectorDiscardingEncryptionEngine<
    SecretKey,
    PlaintextVector,
    CiphertextVector,
>: AbstractEngine where
    SecretKey: GlweSecretKeyEntity,
    PlaintextVector: PlaintextVectorEntity,
    CiphertextVector: GlweCiphertextVectorEntity<KeyFlavor = SecretKey::KeyFlavor>,
{
    /// Encrypts a GLWE ciphertext vector .
    fn discard_encrypt_glwe_ciphertext_vector(
        &mut self,
        key: &SecretKey,
        output: &mut CiphertextVector,
        input: &PlaintextVector,
        noise: Variance,
    ) -> Result<(), GlweCiphertextVectorDiscardingEncryptionError<Self::EngineError>>;

    /// Unsafely encrypts a GLWE ciphertext vector .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextVectorDiscardingEncryptionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_encrypt_glwe_ciphertext_vector_unchecked(
        &mut self,
        key: &SecretKey,
        output: &mut CiphertextVector,
        input: &PlaintextVector,
        noise: Variance,
    );
}
