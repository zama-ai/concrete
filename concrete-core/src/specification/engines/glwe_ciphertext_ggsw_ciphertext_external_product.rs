use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GgswCiphertextEntity, GlweCiphertextEntity};

use super::engine_error;

engine_error! {
    GlweCiphertextGgswCiphertextExternalProductError for
    GlweCiphertextGgswCiphertextExternalProductEngine @
    PolynomialSizeMismatch => "The GGSW ciphertext and GLWE ciphertext polynomial sizes must be the same.",
    GlweDimensionMismatch => "The GGSW ciphertext and GLWE ciphertext GLWE dimension must be the same."
}

impl<EngineError: std::error::Error> GlweCiphertextGgswCiphertextExternalProductError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks<GlweCiphertext, GgswCiphertext>(
        glwe_input: &GlweCiphertext,
        ggsw_input: &GgswCiphertext,
    ) -> Result<(), Self>
    where
        GlweCiphertext: GlweCiphertextEntity,
        GgswCiphertext: GgswCiphertextEntity,
    {
        if glwe_input.polynomial_size().0 != ggsw_input.polynomial_size().0 {
            return Err(Self::PolynomialSizeMismatch);
        }
        if glwe_input.glwe_dimension().0 != ggsw_input.glwe_dimension().0 {
            return Err(Self::GlweDimensionMismatch);
        }
        Ok(())
    }
}

/// A trait for engines computing the external product between a GLWE ciphertext (of dimension
/// 1) and a GSW ciphertext.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a GLWE ciphertext vector containing
/// the result of the external product between a `glwe_input` GLWE ciphertext and a `ggsw_input`
/// GSW ciphertext.
///
/// # Formal Definition
pub trait GlweCiphertextGgswCiphertextExternalProductEngine<GlweInput, GgswInput, Output>:
    AbstractEngine
where
    GlweInput: GlweCiphertextEntity,
    GgswInput: GgswCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
    Output: GlweCiphertextEntity<KeyDistribution = GlweInput::KeyDistribution>,
{
    /// Computes the external product between a GLWE and a GSW ciphertext.
    fn compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweInput,
        ggsw_input: &GgswInput,
    ) -> Result<Output, GlweCiphertextGgswCiphertextExternalProductError<Self::EngineError>>;

    /// Unsafely computes the external product between a GLWE and a GSW ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextGgswCiphertextExternalProductError`]. For safety concerns _specific_ to
    /// an engine, refer to the implementer safety section.
    unsafe fn compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweInput,
        ggsw_input: &GgswInput,
    ) -> Output;
}
