use super::engine_error;
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::{
    GlweCiphertextVectorEntity, LweBootstrapKeyEntity, LweCiphertextVectorEntity,
};

engine_error! {
    LweCiphertextVectorDiscardingBootstrapError for LweCiphertextVectorDiscardingBootstrapEngine @
    InputLweDimensionMismatch => "The input vector and key input LWE dimension must be the same.",
    OutputLweDimensionMismatch => "The output vector and key output LWE dimension must be the same.",
    AccumulatorGlweDimensionMismatch => "The accumulator vector and key GLWE dimension must be the same.",
    AccumulatorPolynomialSizeMismatch => "The accumulator vector and key polynomial size must be the same.",
    AccumulatorCountMismatch => "The accumulator count and input ciphertext count must be the same.",
    CiphertextCountMismatch => "The input and output ciphertext count must be the same."
}

/// A trait for engines bootstrapping (discarding) LWE ciphertext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext vector
/// with the element-wise bootstrap of the `input` LWE ciphertext vector, using the `acc`
/// accumulator as lookup-table, and the `bsk` bootstrap key.
///
/// # Formal Definition
pub trait LweCiphertextVectorDiscardingBootstrapEngine<
    BootstrapKey,
    AccumulatorVector,
    InputCiphertextVector,
    OutputCiphertextVector,
>: AbstractEngine where
    BootstrapKey: LweBootstrapKeyEntity,
    AccumulatorVector: GlweCiphertextVectorEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
    InputCiphertextVector: LweCiphertextVectorEntity<KeyDistribution = BootstrapKey::InputKeyDistribution>,
    OutputCiphertextVector: LweCiphertextVectorEntity<KeyDistribution = BootstrapKey::OutputKeyDistribution>,
{
    /// Bootstraps an LWE ciphertext vector.
    fn discard_bootstrap_lwe_ciphertext_vector(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
        acc: &AccumulatorVector,
        bsk: &BootstrapKey,
    ) -> Result<(), LweCiphertextVectorDiscardingBootstrapError<Self::EngineError>>;

    /// Unsafely bootstraps an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorDiscardingBootstrapError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_bootstrap_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
        acc: &AccumulatorVector,
        bsk: &BootstrapKey,
    );
}
