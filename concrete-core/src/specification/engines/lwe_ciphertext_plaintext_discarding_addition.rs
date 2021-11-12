use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, PlaintextEntity};

engine_error! {
    LweCiphertextPlaintextDiscardingAdditionError for LweCiphertextPlaintextDiscardingAdditionEngine @
    LweDimensionMismatch => "The input and output ciphertext LWE dimensions must be the same."
}

/// A trait for engines adding (discarding) plaintext to LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// addition of the `input_1` LWE ciphertext with the `input_2` plaintext.
///
/// # Formal Definition
pub trait LweCiphertextPlaintextDiscardingAdditionEngine<
    InputCiphertext,
    Plaintext,
    OutputCiphertext,
>: AbstractEngine where
    Plaintext: PlaintextEntity,
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyFlavor = InputCiphertext::KeyFlavor>,
{
    /// Adds a plaintext to an LWE ciphertext.
    fn discard_add_lwe_ciphertext_plaintext(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &Plaintext,
    ) -> Result<(), LweCiphertextPlaintextDiscardingAdditionError<Self::EngineError>>;

    /// Unsafely adds a plaintext to an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextPlaintextDiscardingAdditionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_add_lwe_ciphertext_plaintext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &Plaintext,
    );
}
