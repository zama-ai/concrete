use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, LweCiphertextVectorEntity};
use concrete_commons::parameters::LweCiphertextIndex;

engine_error! {
    LweCiphertextDiscardingStoringError for LweCiphertextDiscardingStoringEngine @
    LweDimensionMismatch => "The input and output LWE dimensions must be the same.",
    IndexTooLarge => "The index must not exceed the size of the vector."
}

/// A trait for engines storing (discarding) LWE ciphertexts in LWE ciphertext vectors.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `i`th LWE ciphertext of the
/// `vector` LWE ciphertext vector, with the `ciphertext` LWE ciphertext.
///
/// # Formal Definition
pub trait LweCiphertextDiscardingStoringEngine<Ciphertext, CiphertextVector>:
    AbstractEngine
where
    CiphertextVector: LweCiphertextVectorEntity,
    Ciphertext: LweCiphertextEntity<KeyFlavor = CiphertextVector::KeyFlavor>,
{
    /// Stores an LWE ciphertext in an LWE ciphertext vector.
    fn discard_store_lwe_ciphertext(
        &mut self,
        vector: &mut CiphertextVector,
        ciphertext: &Ciphertext,
        i: LweCiphertextIndex,
    ) -> Result<(), LweCiphertextDiscardingStoringError<Self::EngineError>>;

    /// Unsafely stores an LWE ciphertext in a LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingStoringError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_store_lwe_ciphertext_unchecked(
        &mut self,
        vector: &mut CiphertextVector,
        ciphertext: &Ciphertext,
        i: LweCiphertextIndex,
    );
}
