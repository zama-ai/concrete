use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextEntity;

engine_error! {
    LweCiphertextDiscardingOppositeError for LweCiphertextDiscardingOppositeEngine @
    LweDimensionMismatch => "The input and output LWE dimension must be the same."
}

impl<EngineError: std::error::Error> LweCiphertextDiscardingOppositeError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks<InputCiphertext, OutputCiphertext>(
        output: &OutputCiphertext,
        input: &InputCiphertext,
    ) -> Result<(), Self>
    where
        InputCiphertext: LweCiphertextEntity,
        OutputCiphertext: LweCiphertextEntity<KeyDistribution = InputCiphertext::KeyDistribution>,
    {
        if input.lwe_dimension() != output.lwe_dimension() {
            return Err(Self::LweDimensionMismatch);
        }

        Ok(())
    }
}

/// A trait for engines computing the opposite (discarding) LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with
/// the opposite of the `input` LWE ciphertext.
///
/// # Formal Definition
pub trait LweCiphertextDiscardingOppositeEngine<InputCiphertext, OutputCiphertext>:
    AbstractEngine
where
    InputCiphertext: LweCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity<KeyDistribution = InputCiphertext::KeyDistribution>,
{
    /// Computes the opposite of an LWE ciphertext.
    fn discard_opp_lwe_ciphertext(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
    ) -> Result<(), LweCiphertextDiscardingOppositeError<Self::EngineError>>;

    /// Unsafely computes the opposite of an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingOppositeError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_opp_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
    );
}
