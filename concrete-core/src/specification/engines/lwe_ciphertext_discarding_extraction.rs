use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, LweCiphertextEntity};
use concrete_commons::parameters::MonomialDegree;

engine_error! {
    LweCiphertextDiscardingExtractionError for LweCiphertextDiscardingExtractionEngine @
    SizeMismatch => "The sizes of the output LWE (LWE dimension) and the input GLWE (GLWE \
                     dimension * poly size) must be compatible.",
    MonomialDegreeTooLarge => "The monomial degree must be lower than the GLWE polynomial size."
}

/// A trait for engines extracting (discarding) LWE ciphertext from GLWE ciphertexts.
///
/// # Semantics
///
/// This [discarding](super#operation-semantics) operation fills the `output` LWE ciphertext with the
/// extraction of the `nth` coefficient of the `input` GLWE ciphertext.
///
/// # Formal definition
///
/// This operation is usually referred to as a _sample extract_ in the literature.
pub trait LweCiphertextDiscardingExtractionEngine<GlweCiphertext, LweCiphertext>:
    AbstractEngine
where
    GlweCiphertext: GlweCiphertextEntity,
    LweCiphertext: LweCiphertextEntity<KeyFlavor = GlweCiphertext::KeyFlavor>,
{
    /// Extracts an LWE ciphertext from a GLWE ciphertext.
    fn discard_extract_lwe_ciphertext(
        &mut self,
        output: &mut LweCiphertext,
        input: &GlweCiphertext,
        nth: MonomialDegree,
    ) -> Result<(), LweCiphertextDiscardingExtractionError<Self::EngineError>>;

    /// Unsafely extracts an LWE ciphertext from a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextDiscardingExtractionError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn discard_extract_lwe_ciphertext_unchecked(
        &mut self,
        output: &mut LweCiphertext,
        input: &GlweCiphertext,
        nth: MonomialDegree,
    );
}
