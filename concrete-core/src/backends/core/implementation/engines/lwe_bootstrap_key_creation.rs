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

impl LweBootstrapKeyCreationEngine<LweSecretKey32, GlweSecretKey32, LweBootstrapKey32>
    for CoreEngine
{
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<LweBootstrapKey32, LweBootstrapKeyCreationError<Self::EngineError>> {
        if decomposition_base_log.0 == 0 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionBaseLog);
        }
        if decomposition_level_count.0 == 1 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionLevelCount);
        }
        if decomposition_base_log.0 * decomposition_level_count.0 > 32 {
            return Err(LweBootstrapKeyCreationError::DecompositionTooLarge);
        }
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

impl LweBootstrapKeyCreationEngine<LweSecretKey64, GlweSecretKey64, LweBootstrapKey64>
    for CoreEngine
{
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<LweBootstrapKey64, LweBootstrapKeyCreationError<Self::EngineError>> {
        if decomposition_base_log.0 == 0 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionBaseLog);
        }
        if decomposition_level_count.0 == 1 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionLevelCount);
        }
        if decomposition_base_log.0 * decomposition_level_count.0 > 64 {
            return Err(LweBootstrapKeyCreationError::DecompositionTooLarge);
        }
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

impl LweBootstrapKeyCreationEngine<LweSecretKey32, GlweSecretKey32, FourierLweBootstrapKey32>
    for CoreEngine
{
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<FourierLweBootstrapKey32, LweBootstrapKeyCreationError<Self::EngineError>> {
        if decomposition_base_log.0 == 0 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionBaseLog);
        }
        if decomposition_level_count.0 == 1 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionLevelCount);
        }
        if decomposition_base_log.0 * decomposition_level_count.0 > 32 {
            return Err(LweBootstrapKeyCreationError::DecompositionTooLarge);
        }
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
        let mut fourier_key = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        fourier_key.fill_with_forward_fourier(&key);
        FourierLweBootstrapKey32(fourier_key)
    }
}

impl LweBootstrapKeyCreationEngine<LweSecretKey64, GlweSecretKey64, FourierLweBootstrapKey64>
    for CoreEngine
{
    fn create_lwe_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<FourierLweBootstrapKey64, LweBootstrapKeyCreationError<Self::EngineError>> {
        if decomposition_base_log.0 == 0 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionBaseLog);
        }
        if decomposition_level_count.0 == 1 {
            return Err(LweBootstrapKeyCreationError::NullDecompositionLevelCount);
        }
        if decomposition_base_log.0 * decomposition_level_count.0 > 32 {
            return Err(LweBootstrapKeyCreationError::DecompositionTooLarge);
        }
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
        let mut fourier_key = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            output_key.0.key_size().to_glwe_size(),
            output_key.0.polynomial_size(),
            decomposition_level_count,
            decomposition_base_log,
            input_key.0.key_size(),
        );
        fourier_key.fill_with_forward_fourier(&key);
        FourierLweBootstrapKey64(fourier_key)
    }
}
