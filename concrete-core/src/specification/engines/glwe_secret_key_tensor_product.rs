use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweSecretKeyEntity;

engine_error! {
    GlweKeyTensorProductError for GlweKeyTensorProductEngine @
    PolynomialSizeMismatch => "The polynomial size of the two input keys is not the same",
    InputGlweDimensionMismatch => "The Glwe dimension of the two input keys is not the same"
}

impl<EngineError: std::error::Error> GlweKeyTensorProductError<EngineError> {
    pub fn perform_generic_checks<InputKey1, InputKey2>(
        input1: &InputKey1,
        input2: &InputKey2,
    ) -> Result<(), Self>
    where
        InputKey1: GlweSecretKeyEntity,
        InputKey2: GlweSecretKeyEntity<KeyDistribution = InputKey1::KeyDistribution>,
    {
        if input1.polynomial_size().0 != input2.polynomial_size().0 {
            return Err(Self::PolynomialSizeMismatch);
        }
        if input1.glwe_dimension().0 != input2.glwe_dimension().0 {
            return Err(Self::InputGlweDimensionMismatch);
        }
        Ok(())
    }
}
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates the tensor product of the two
/// input GLWE secret keys `input1` and `input2`.
///
/// # Formal Definition
///
/// The goal of this function is to take as input two GLWE secret keys s1 and s2, and
/// create their tensor product s1 x s2
pub trait GlweKeyTensorProductEngine<InputKey1, InputKey2>: AbstractEngine
where
    InputKey1: GlweSecretKeyEntity,
    InputKey2: GlweSecretKeyEntity<KeyDistribution = InputKey1::KeyDistribution>,
{
    fn tensor_product_glwe_secret_key(
        &mut self,
        input1: InputKey1,
        input2: InputKey2,
    ) -> Result<(), GlweKeyTensorProductError<Self::EngineError>>;

    /// Unsafely performs a tensor product of two GLWE secret keys.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweKeyTensorProductError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.

    unsafe fn tensor_product_glwe_secret_key_unchecked(
        &mut self,
        input1: InputKey1,
        input2: InputKey2,
    );
}
