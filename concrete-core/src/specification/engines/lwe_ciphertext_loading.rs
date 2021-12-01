use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{LweCiphertextEntity, LweCiphertextVectorEntity};
use concrete_commons::parameters::LweCiphertextIndex;

engine_error! {
    LweCiphertextLoadingError for LweCiphertextLoadingEngine @
    IndexTooLarge => "The index must not exceed the size of the vector."
}

/// A trait for engines loading LWE ciphertexts from LWE ciphertext vectors.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates an LWE ciphertext containing the
/// `i`th LWE ciphertext of the `vector` LWE ciphertext vector.
///
/// # Formal Definition
pub trait LweCiphertextLoadingEngine<CiphertextVector, Ciphertext>: AbstractEngine
where
    Ciphertext: LweCiphertextEntity,
    CiphertextVector: LweCiphertextVectorEntity<KeyFlavor = Ciphertext::KeyFlavor>,
{
    /// Loads an LWE ciphertext from an LWE ciphertext vector.
    fn load_lwe_ciphertext(
        &mut self,
        vector: &CiphertextVector,
        i: LweCiphertextIndex,
    ) -> Result<Ciphertext, LweCiphertextLoadingError<Self::EngineError>>;

    /// Unsafely loads an LWE ciphertext from an LWE ciphertext vector.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextLoadingError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn load_lwe_ciphertext_unchecked(
        &mut self,
        vector: &CiphertextVector,
        i: LweCiphertextIndex,
    ) -> Ciphertext;
}
