use super::engine_error;
use crate::prelude::PlaintextVectorEntity;
use concrete_commons::parameters::GlweSize;

use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweCiphertextEntity;

engine_error! {
    GlweCiphertextTrivialEncryptionError for GlweCiphertextTrivialEncryptionEngine @
}

/// A trait for engines trivially encrypting GLWE ciphertext.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext containing the
/// trivial encryption of the `input` plaintext vector with the requested `glwe_size`.
///
/// # Formal Definition
///
/// A trivial encryption uses a zero mask and no noise.
/// It is absolutely not secure, as the body contains a direct copy of the plaintext
pub trait GlweCiphertextTrivialEncryptionEngine<PlaintextVector, Ciphertext>:
    AbstractEngine
where
    PlaintextVector: PlaintextVectorEntity,
    Ciphertext: GlweCiphertextEntity,
{
    /// Trivially encrypts a plaintext vector into a GLWE ciphertext.
    fn trivially_encrypt_glwe_ciphertext(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector,
    ) -> Result<Ciphertext, GlweCiphertextTrivialEncryptionError<Self::EngineError>>;

    /// Unsafely creates the trivial GLWE encryption of the plaintext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextTrivialEncryptionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn trivially_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector,
    ) -> Ciphertext;
}
