use crate::backends::core::engines::CoreEngine;
use crate::backends::core::entities::{
    GlweSecretKey32, GlweSecretKey64, LweSecretKey32, LweSecretKey64,
};
use crate::specification::engines::{
    GlweToLweSecretKeyTransmutationEngine, GlweToLweSecretKeyTransmutationEngineError,
};

impl GlweToLweSecretKeyTransmutationEngine<GlweSecretKey32, LweSecretKey32> for CoreEngine {
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use concrete_commons::parameters::{GlweDimension, LweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    ///
    /// let mut engine = CoreEngine::new()?;
    ///
    /// let glwe_secret_key: GlweSecretKey32 =
    ///     engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// assert_eq!(glwe_secret_key.glwe_dimension(), glwe_dimension);
    /// assert_eq!(glwe_secret_key.polynomial_size(), polynomial_size);
    ///
    /// let lwe_secret_key = engine.transmute_glwe_secret_key_to_lwe_secret_key(glwe_secret_key)?;
    /// assert_eq!(lwe_secret_key.lwe_dimension(), LweDimension(8));
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn transmute_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        glwe_secret_key: GlweSecretKey32,
    ) -> Result<LweSecretKey32, GlweToLweSecretKeyTransmutationEngineError<Self::EngineError>> {
        Ok(unsafe { self.transmute_glwe_secret_key_to_lwe_secret_key_unchecked(glwe_secret_key) })
    }

    unsafe fn transmute_glwe_secret_key_to_lwe_secret_key_unchecked(
        &mut self,
        glwe_secret_key: GlweSecretKey32,
    ) -> LweSecretKey32 {
        LweSecretKey32(glwe_secret_key.0.into_lwe_secret_key())
    }
}

impl GlweToLweSecretKeyTransmutationEngine<GlweSecretKey64, LweSecretKey64> for CoreEngine {
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use concrete_commons::parameters::{GlweDimension, LweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    ///
    /// let mut engine = CoreEngine::new()?;
    ///
    /// let glwe_secret_key: GlweSecretKey64 =
    ///     engine.create_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// assert_eq!(glwe_secret_key.glwe_dimension(), glwe_dimension);
    /// assert_eq!(glwe_secret_key.polynomial_size(), polynomial_size);
    ///
    /// let lwe_secret_key = engine.transmute_glwe_secret_key_to_lwe_secret_key(glwe_secret_key)?;
    /// assert_eq!(lwe_secret_key.lwe_dimension(), LweDimension(8));
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn transmute_glwe_secret_key_to_lwe_secret_key(
        &mut self,
        glwe_secret_key: GlweSecretKey64,
    ) -> Result<LweSecretKey64, GlweToLweSecretKeyTransmutationEngineError<Self::EngineError>> {
        Ok(unsafe { self.transmute_glwe_secret_key_to_lwe_secret_key_unchecked(glwe_secret_key) })
    }

    unsafe fn transmute_glwe_secret_key_to_lwe_secret_key_unchecked(
        &mut self,
        glwe_secret_key: GlweSecretKey64,
    ) -> LweSecretKey64 {
        LweSecretKey64(glwe_secret_key.0.into_lwe_secret_key())
    }
}
