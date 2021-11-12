use concrete_commons::parameters::{GlweDimension, PolynomialSize};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{GlweSecretKey32, GlweSecretKey64};
use crate::backends::core::private::crypto::secret::GlweSecretKey as ImplGlweSecretKey;
use crate::specification::engines::{GlweSecretKeyCreationEngine, GlweSecretKeyCreationError};

impl GlweSecretKeyCreationEngine<GlweSecretKey32> for CoreEngine {
    fn create_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweSecretKey32, GlweSecretKeyCreationError<Self::EngineError>> {
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

impl GlweSecretKeyCreationEngine<GlweSecretKey64> for CoreEngine {
    fn create_glwe_secret_key(
        &mut self,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweSecretKey64, GlweSecretKeyCreationError<Self::EngineError>> {
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
