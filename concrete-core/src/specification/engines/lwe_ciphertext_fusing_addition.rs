use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextEntity;

engine_error! {
    LweCiphertextFusingAdditionError for LweCiphertextFusingAdditionEngine @
    LweDimensionMismatch => "The input and output LWE dimensions must be the same."
}

/// A trait for engines adding (fusing) LWE ciphertexts.
///
/// # Semantics
///
/// This [fusing](super#operation-semantics) operation adds the `input` LWE ciphertext to the
/// `output` LWE ciphertext.
///
/// # Formal Definition
pub trait LweCiphertextFusingAdditionEngine<InputCiphertext, OutputCiphertext>:
    AbstractEngine
where
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyFlavor = InputCiphertext::KeyFlavor>,
{
    /// Adds an LWE ciphertext to an other.
    fn fuse_add_lwe_ciphertext(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
    ) -> Result<(), LweCiphertextFusingAdditionError<Self::EngineError>>;

    /// Unsafely add an LWE ciphertext to an other.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextFusingAdditionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn fuse_add_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
    );
}
