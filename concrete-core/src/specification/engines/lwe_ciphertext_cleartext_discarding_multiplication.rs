use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{CleartextEntity, LweCiphertextEntity};

engine_error! {
    LweCiphertextCleartextDiscardingMultiplicationError for LweCiphertextCleartextDiscardingMultiplicationEngine @
    LweDimensionMismatch => "The input and output ciphertext LWE dimension must be the same."
}

impl<EngineError: std::error::Error>
    LweCiphertextCleartextDiscardingMultiplicationError<EngineError>
{
    /// Validates the inputs
    pub fn perform_generic_checks<InputCiphertext, OutputCiphertext>(
        output: &OutputCiphertext,
        input_1: &InputCiphertext,
    ) -> Result<(), Self>
    where
        InputCiphertext: LweCiphertextEntity,
        OutputCiphertext: LweCiphertextEntity<KeyDistribution = InputCiphertext::KeyDistribution>,
    {
        if output.lwe_dimension() != input_1.lwe_dimension() {
            return Err(Self::LweDimensionMismatch);
        }
        Ok(())
    }
}

/// A trait for engines multiplying (discarding) LWE ciphertext by cleartexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with
/// the multiplication of the `input_1` LWE ciphertext with the `input_2` cleartext.
///
/// # Formal Definition
pub trait LweCiphertextCleartextDiscardingMultiplicationEngine<
    InputCiphertext,
    Cleartext,
    OutputCiphertext,
>: AbstractEngine where
    Cleartext: CleartextEntity,
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyDistribution = InputCiphertext::KeyDistribution>,
{
    /// Multiply an LWE ciphertext with a cleartext.
    fn discard_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &Cleartext,
    ) -> Result<(), LweCiphertextCleartextDiscardingMultiplicationError<Self::EngineError>>;

    /// Unsafely multiply an LWE ciphertext with a cleartext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextCleartextDiscardingMultiplicationError`]. For safety concerns _specific_
    /// to an engine, refer to the implementer safety section.
    unsafe fn discard_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input_1: &InputCiphertext,
        input_2: &Cleartext,
    );
}
