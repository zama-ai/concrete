use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextVectorEntity;

engine_error! {
    LweCiphertextVectorFusingNegationError for LweCiphertextVectorFusingNegationEngine @
}

/// A trait for engines negating (fusing) LWE ciphertext vectors.
///
/// # Semantics
///
/// This [fusing](super#operation-semantics) operation negates the `input` LWE ciphertext vector.
///
///  # Formal Definition
pub trait LweCiphertextVectorFusingNegationEngine<CiphertextVector>: AbstractEngine
where
    CiphertextVector: LweCiphertextVectorEntity,
{
    /// Negates an LWE ciphertext vector.
    fn fuse_neg_lwe_ciphertext_vector(
        &mut self,
        input: &mut CiphertextVector,
    ) -> Result<(), LweCiphertextVectorFusingNegationError<Self::EngineError>>;

    /// Unsafely negates an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextVectorFusingNegationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn fuse_neg_lwe_ciphertext_vector_unchecked(&mut self, input: &mut CiphertextVector);
}
