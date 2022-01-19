use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::GlweSecretKeyEntity;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};

engine_error! {
    GlweSecretKeyCreationError for GlweSecretKeyCreationEngine @
    NullGlweDimension => "The secret key GLWE dimension must be greater than zero.",
    NullPolynomialSize => "The secret key polynomial size must be greater than zero.",
    SizeOnePolynomial => "The secret key polynomial size must be greater than one. Otherwise you \
                          should prefer the LWE scheme."
}

impl<EngineError: std::error::Error> GlweSecretKeyCreationError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks(
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<(), Self> {
        if glwe_dimension.0 == 0 {
            return Err(Self::NullGlweDimension);
        }

        if polynomial_size.0 == 0 {
            return Err(Self::NullPolynomialSize);
        }

        if polynomial_size.0 == 1 {
            return Err(Self::SizeOnePolynomial);
        }

        Ok(())
    }
}

/// A trait for engines creating GLWE secret keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates a fresh GLWE secret key.
///
/// # Formal Definition
pub trait GlweSecretKeyCreationEngine<SecretKey>: AbstractEngine
where
    SecretKey: GlweSecretKeyEntity,
{
    /// Creates a new GLWE secret key.
    fn create_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<SecretKey, GlweSecretKeyCreationError<Self::EngineError>>;

    /// Unsafely creates a new GLWE secret key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweSecretKeyCreationError`]. For safety concerns _specific_ to an engine, refer to
    /// the implementer safety section.
    unsafe fn create_glwe_secret_key_unchecked(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> SecretKey;
}
