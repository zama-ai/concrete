use super::engine_error;
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::{
    GlweCiphertextEntity, LweBootstrapKeyEntity, LweCiphertextEntity,
};

engine_error! {
    LweCiphertextDiscardingBootstrapError for LweCiphertextDiscardingBootstrapEngine @
    InputLweDimensionMismatch => "The input ciphertext and key LWE dimension must be the same.",
    OutputLweDimensionMismatch => "The output ciphertext dimension and key size (dimension * \
                                   polynomial size) must be the same.",
    AccumulatorPolynomialSizeMismatch => "The accumulator and key polynomial sizes must be the same.",
    AccumulatorGlweDimensionMismatch => "The accumulator and key GLWE dimensions must be the same."
}

/// A trait for engines bootstrapping (discarding) LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// bootstrap of the `input` LWE ciphertext, using the `acc` accumulator as lookup-table, and the
/// `bsk` bootstrap key.
///
/// # Formal Definition
pub trait LweCiphertextDiscardingBootstrapEngine<
    BootstrapKey,
    Accumulator,
    InputCiphertext,
    OutputCiphertext,
>: AbstractEngine where
    BootstrapKey: LweBootstrapKeyEntity,
    Accumulator: GlweCiphertextEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
    InputCiphertext: LweCiphertextEntity<KeyDistribution = BootstrapKey::InputKeyDistribution>,
    OutputCiphertext: LweCiphertextEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
{
    /// Bootstrap an LWE ciphertext .
    fn discard_bootstrap_lwe_ciphertext(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
        acc: &Accumulator,
        bsk: &BootstrapKey,
    ) -> Result<(), LweCiphertextDiscardingBootstrapError<Self::EngineError>>;

    /// Unsafely bootstrap an LWE ciphertext .
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingBootstrapError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_bootstrap_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        input: &InputCiphertext,
        acc: &Accumulator,
        bsk: &BootstrapKey,
    );
}
