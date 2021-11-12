use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextEntity;

engine_error! {
    LweCiphertextDiscardingAdditionError for LweCiphertextDiscardingAdditionEngine @
    LweDimensionMismatch => "All the ciphertext LWE dimensions must be the same."
}

/// A trait for engines adding (discarding) LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// addition of the `input_1` LWE ciphertext and the `input_2` LWE ciphertext.
///
/// # Formal Definition
pub trait LweCiphertextDiscardingAdditionEngine<InputCiphertext, OutputCiphertext>:
    AbstractEngine
where
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyFlavor = InputCiphertext::KeyFlavor>,
{
    /// Adds two LWE ciphertexts.
    fn discard_add_lwe_ciphertext(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &InputCiphertext,
    ) -> Result<(), LweCiphertextDiscardingAdditionError<Self::EngineError>>;

    /// Unsafely adds two LWE ciphertexts.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingAdditionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_add_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &InputCiphertext,
    );
}
