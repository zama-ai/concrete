use concrete_commons::parameters::{GlweDimension, PolynomialSize};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{GlweSecretKey32, GlweSecretKey64};
use crate::backends::core::private::crypto::secret::GlweSecretKey as ImplGlweSecretKey;
use crate::specification::engines::{GlweSecretKeyCreationEngine, GlweSecretKeyCreationError};

/// # Description:
/// Implementation of [`GlweSecretKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl GlweSecretKeyCreationEngine<GlweSecretKey32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let glwe_secret_key: GlweSecretKey32 =
    ///     engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// #
    /// assert_eq!(glwe_secret_key.glwe_dimension(), glwe_dimension);
    /// assert_eq!(glwe_secret_key.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(glwe_secret_key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweSecretKey32, GlweSecretKeyCreationError<Self::EngineError>> {
        GlweSecretKeyCreationError::perform_generic_checks(glwe_dimension, polynomial_size)?;
        Ok(unsafe { self.create_glwe_secret_key_unchecked(glwe_dimension, polynomial_size) })
    }

    unsafe fn create_glwe_secret_key_unchecked(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> GlweSecretKey32 {
        GlweSecretKey32(ImplGlweSecretKey::generate_binary(
            glwe_dimension,
            polynomial_size,
            &mut self.secret_generator,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweSecretKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl GlweSecretKeyCreationEngine<GlweSecretKey64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let glwe_secret_key: GlweSecretKey64 =
    ///     engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// #
    /// assert_eq!(glwe_secret_key.glwe_dimension(), glwe_dimension);
    /// assert_eq!(glwe_secret_key.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(glwe_secret_key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweSecretKey64, GlweSecretKeyCreationError<Self::EngineError>> {
        GlweSecretKeyCreationError::perform_generic_checks(glwe_dimension, polynomial_size)?;
        Ok(unsafe { self.create_glwe_secret_key_unchecked(glwe_dimension, polynomial_size) })
    }

    unsafe fn create_glwe_secret_key_unchecked(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> GlweSecretKey64 {
        GlweSecretKey64(ImplGlweSecretKey::generate_binary(
            glwe_dimension,
            polynomial_size,
            &mut self.secret_generator,
        ))
    }
}
