use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierLweBootstrapKey32, FourierLweBootstrapKey64, GlweSecretKey32, GlweSecretKey64,
    LweBootstrapKey32, LweBootstrapKey64, LweSecretKey32, LweSecretKey64,
};
use crate::backends::core::private::crypto::bootstrap::{
    FourierBootstrapKey as ImplFourierBootstrapKey,
    StandardBootstrapKey as ImplStandardBootstrapKey,
};
use crate::backends::core::private::math::fft::Complex64;
use crate::specification::engines::{LweBootstrapKeyCreationEngine, LweBootstrapKeyCreationError};

/// # Description:
/// Implementation of [`LweBootstrapKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers. It outputs a bootstrap key in the standard domain.
impl LweBootstrapKeyCreationEngine<LweSecretKey32, GlweSecretKey32, LweBootstrapKey32>
    for CoreEngine
{
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(256));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    ///
    /// let bsk: LweBootstrapKey32 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// #
    /// assert_eq!(bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(bsk.polynomial_size(), poly_size);
    /// assert_eq!(bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(bsk.decomposition_level_count(), dec_lc);
    ///
    /// engine.destroy(lwe_sk)?;
    /// engine.destroy(glwe_sk)?;
    /// engine.destroy(bsk)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<LweBootstrapKey32, LweBootstrapKeyCreationError<Self::EngineError>> {
        LweBootstrapKeyCreationError::perform_generic_checks(
            decomposition_base_log,
            decomposition_level_count,
            32,
        )?;
        Ok(unsafe {
            self.create_lwe_bootstrap_key_unchecked(
                input_key,
                output_key,
                decomposition_base_log,
                decomposition_level_count,
                noise,
            )
        })
    }

    unsafe fn create_lwe_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> LweBootstrapKey32 {
        let mut key = ImplStandardBootstrapKey::allocate(
            0,
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        key.fill_with_new_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        LweBootstrapKey32(key)
    }
}

/// # Description:
/// Implementation of [`LweBootstrapKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers. It outputs a bootstrap key in the standard domain.
impl LweBootstrapKeyCreationEngine<LweSecretKey64, GlweSecretKey64, LweBootstrapKey64>
    for CoreEngine
{
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(256));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    ///
    /// let bsk: LweBootstrapKey64 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// #
    /// assert_eq!(bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(bsk.polynomial_size(), poly_size);
    /// assert_eq!(bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(bsk.decomposition_level_count(), dec_lc);
    ///
    /// engine.destroy(lwe_sk)?;
    /// engine.destroy(glwe_sk)?;
    /// engine.destroy(bsk)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<LweBootstrapKey64, LweBootstrapKeyCreationError<Self::EngineError>> {
        LweBootstrapKeyCreationError::perform_generic_checks(
            decomposition_base_log,
            decomposition_level_count,
            64,
        )?;
        Ok(unsafe {
            self.create_lwe_bootstrap_key_unchecked(
                input_key,
                output_key,
                decomposition_base_log,
                decomposition_level_count,
                noise,
            )
        })
    }

    unsafe fn create_lwe_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> LweBootstrapKey64 {
        let mut key = ImplStandardBootstrapKey::allocate(
            0,
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        key.fill_with_new_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        LweBootstrapKey64(key)
    }
}

/// # Description:
/// Implementation of [`LweBootstrapKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers. It outputs a bootstrap key in the Fourier domain.
impl LweBootstrapKeyCreationEngine<LweSecretKey32, GlweSecretKey32, FourierLweBootstrapKey32>
    for CoreEngine
{
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(256));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey32 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    ///
    /// let bsk: FourierLweBootstrapKey32 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// #
    /// assert_eq!(bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(bsk.polynomial_size(), poly_size);
    /// assert_eq!(bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(bsk.decomposition_level_count(), dec_lc);
    ///
    /// engine.destroy(lwe_sk)?;
    /// engine.destroy(glwe_sk)?;
    /// engine.destroy(bsk)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<FourierLweBootstrapKey32, LweBootstrapKeyCreationError<Self::EngineError>> {
        LweBootstrapKeyCreationError::perform_generic_checks(
            decomposition_base_log,
            decomposition_level_count,
            32,
        )?;
        Ok(unsafe {
            self.create_lwe_bootstrap_key_unchecked(
                input_key,
                output_key,
                decomposition_base_log,
                decomposition_level_count,
                noise,
            )
        })
    }

    unsafe fn create_lwe_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> FourierLweBootstrapKey32 {
        let mut key = ImplStandardBootstrapKey::allocate(
            0,
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        key.fill_with_new_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        let fourier_key = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );

        let mut fourier_key = FourierLweBootstrapKey32(fourier_key);
        let buffers = self.get_fourier_bootstrap_u32_buffer(&fourier_key);
        fourier_key.0.fill_with_forward_fourier(&key, buffers);
        fourier_key
    }
}

/// # Description:
/// Implementation of [`LweBootstrapKeyCreationEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers. It outputs a bootstrap key in the Fourier domain.
impl LweBootstrapKeyCreationEngine<LweSecretKey64, GlweSecretKey64, FourierLweBootstrapKey64>
    for CoreEngine
{
    /// # Example
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(256));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let lwe_sk: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dim)?;
    /// let glwe_sk: GlweSecretKey64 = engine.create_glwe_secret_key(glwe_dim, poly_size)?;
    ///
    /// let bsk: FourierLweBootstrapKey64 =
    ///     engine.create_lwe_bootstrap_key(&lwe_sk, &glwe_sk, dec_bl, dec_lc, noise)?;
    /// #
    /// assert_eq!(bsk.glwe_dimension(), glwe_dim);
    /// assert_eq!(bsk.polynomial_size(), poly_size);
    /// assert_eq!(bsk.input_lwe_dimension(), lwe_dim);
    /// assert_eq!(bsk.decomposition_base_log(), dec_bl);
    /// assert_eq!(bsk.decomposition_level_count(), dec_lc);
    ///
    /// engine.destroy(lwe_sk);
    /// engine.destroy(glwe_sk);
    /// engine.destroy(bsk);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<FourierLweBootstrapKey64, LweBootstrapKeyCreationError<Self::EngineError>> {
        LweBootstrapKeyCreationError::perform_generic_checks(
            decomposition_base_log,
            decomposition_level_count,
            64,
        )?;
        Ok(unsafe {
            self.create_lwe_bootstrap_key_unchecked(
                input_key,
                output_key,
                decomposition_base_log,
                decomposition_level_count,
                noise,
            )
        })
    }

    unsafe fn create_lwe_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> FourierLweBootstrapKey64 {
        let mut key = ImplStandardBootstrapKey::allocate(
            0,
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        key.fill_with_new_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        let fourier_key = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );

        let mut fourier_key = FourierLweBootstrapKey64(fourier_key);
        let buffers = self.get_fourier_bootstrap_u64_buffer(&fourier_key);
        fourier_key.0.fill_with_forward_fourier(&key, buffers);
        fourier_key
    }
}
