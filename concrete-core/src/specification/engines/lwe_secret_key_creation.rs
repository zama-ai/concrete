use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweSecretKeyEntity;
use concrete_commons::parameters::LweDimension;

engine_error! {
    LweSecretKeyCreationError for LweSecretKeyCreationEngine @
    NullLweDimension => "The LWE dimension must be greater than zero."
}

impl<EngineError: std::error::Error> LweSecretKeyCreationError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks(lwe_dimension: LweDimension) -> Result<(), Self> {
        if lwe_dimension.0 == 0 {
            return Err(Self::NullLweDimension);
        }
        Ok(())
    }
}

/// A trait for engines creating LWE secret keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates a fresh LWE secret key.
///
/// # Formal Definition
pub trait LweSecretKeyCreationEngine<SecretKey>: AbstractEngine
where
    SecretKey: LweSecretKeyEntity,
{
    /// Creates an LWE secret key.
    fn create_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Result<SecretKey, LweSecretKeyCreationError<Self::EngineError>>;

    /// Unsafely creates an LWE secret key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweSecretKeyCreationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn create_lwe_secret_key_unchecked(&mut self, lwe_dimension: LweDimension) -> SecretKey;
}
