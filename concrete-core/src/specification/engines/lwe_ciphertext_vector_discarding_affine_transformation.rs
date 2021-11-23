use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    CleartextVectorEntity, LweCiphertextEntity, LweCiphertextVectorEntity, PlaintextEntity,
};

engine_error! {
    LweCiphertextVectorDiscardingAffineTransformationError for LweCiphertextVectorDiscardingAffineTransformationEngine @
    LweDimensionMismatch => "The output and inputs LWE dimensions must be the same.",
    CleartextCountMismatch => "The cleartext vector count and input vector count must be the same."
}

/// A trait for engines performing (discarding) affine transformation of LWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// result of the affine tranform of the `inputs` LWE ciphertext vector, with the `weights`
/// cleartext vector and the `bias` plaintext.
///
/// # Formal Definition
pub trait LweCiphertextVectorDiscardingAffineTransformationEngine<
    CiphertextVector,
    CleartextVector,
    Plaintext,
    OutputCiphertext,
>: AbstractEngine where
    OutputCiphertext: LweCiphertextEntity,
    CiphertextVector: LweCiphertextVectorEntity<KeyDistribution = OutputCiphertext::KeyDistribution>,
    CleartextVector: CleartextVectorEntity,
    Plaintext: PlaintextEntity,
{
    /// Performs the affine transform of an LWE ciphertext vector.
    fn discard_affine_transform_lwe_ciphertext_vector(
        &mut self,
        output: &mut OutputCiphertext,
        inputs: &CiphertextVector,
        weights: &CleartextVector,
        bias: &Plaintext,
    ) -> Result<(), LweCiphertextVectorDiscardingAffineTransformationError<Self::EngineError>>;

    /// Unsafely performs the affine transform of an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingAffineTransformationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn discard_affine_transform_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut OutputCiphertext,
        inputs: &CiphertextVector,
        weights: &CleartextVector,
        bias: &Plaintext,
    );
}
