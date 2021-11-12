use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextEntity;

engine_error! {
    LweCiphertextFusingNegationError for LweCiphertextFusingNegationEngine @
}

/// A trait for engines negating (fusing) LWE ciphertexts.
///
/// # Semantics
///
/// This [fusing](super#operation-semantics) operation negates the `input` LWE ciphertext.
///
/// # Formal Definition
pub trait LweCiphertextFusingNegationEngine<Ciphertext>: AbstractEngine
where
    Ciphertext: LweCiphertextEntity,
{
    /// Negates an LWE ciphertext.
    fn fuse_neg_lwe_ciphertext(
        &mut self,
        input: &mut Ciphertext,
    ) -> Result<(), LweCiphertextFusingNegationError<Self::EngineError>>;

    /// Unsafely negates an LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextFusingNegationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn fuse_neg_lwe_ciphertext_unchecked(&mut self, input: &mut Ciphertext);
}
