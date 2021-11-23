use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorFusingAdditionError for LweCiphertextVectorFusingAdditionEngine @
    LweDimensionMismatch => "The input and output LWE dimension must be the same.",
    CiphertextCountMismatch => "The input and output vectors length must be the same."
}

/// A trait for engines adding (fusing) LWE ciphertexts vectors.
///
/// # Semantics
///
/// This [fusing](super#operation-semantics) operation adds the `input` LWE ciphertext vector to
/// the `output` LWE ciphertext vector.
///
/// # Formal Definition
pub trait LweCiphertextVectorFusingAdditionEngine<InputCiphertextVector, OutputCiphertextVector>:
    AbstractEngine
where
    InputCiphertextVector: LweCiphertextVectorEntity,
    OutputCiphertextVector: LweCiphertextVectorEntity<KeyDistribution = InputCiphertextVector::KeyDistribution>,
{
    /// Add two LWE ciphertext vectors.
    fn fuse_add_lwe_ciphertext_vector(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
    ) -> Result<(), LweCiphertextVectorFusingAdditionError<Self::EngineError>>;

    /// Unsafely add two LWE ciphertext vectors.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorFusingAdditionError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn fuse_add_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut OutputCiphertextVector,
        input: &InputCiphertextVector,
    );
}
