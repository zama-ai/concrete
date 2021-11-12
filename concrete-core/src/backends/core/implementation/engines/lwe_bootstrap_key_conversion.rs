use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    FourierLweBootstrapKey32, FourierLweBootstrapKey64, LweBootstrapKey32, LweBootstrapKey64,
};
use crate::backends::core::private::crypto::bootstrap::FourierBootstrapKey as ImplFourierBootstrapKey;
use crate::backends::core::private::math::fft::Complex64;
use crate::specification::engines::{
    LweBootstrapKeyConversionEngine, LweBootstrapKeyConversionError,
};
use crate::specification::entities::LweBootstrapKeyEntity;

impl LweBootstrapKeyConversionEngine<LweBootstrapKey32, FourierLweBootstrapKey32> for CoreEngine {
    fn convert_lwe_bootstrap_key(
        &mut self,
        input: &LweBootstrapKey32,
    ) -> Result<FourierLweBootstrapKey32, LweBootstrapKeyConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_lwe_bootstrap_key_unchecked(input) })
    }

    unsafe fn convert_lwe_bootstrap_key_unchecked(
        &mut self,
        input: &LweBootstrapKey32,
    ) -> FourierLweBootstrapKey32 {
        let mut output = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            input.glwe_dimension().to_glwe_size(),
            input.polynomial_size(),
            input.decomposition_level_count(),
            input.decomposition_base_log(),
            input.input_lwe_dimension(),
        );
        output.fill_with_forward_fourier(&input.0);
        FourierLweBootstrapKey32(output)
    }
}

impl LweBootstrapKeyConversionEngine<LweBootstrapKey64, FourierLweBootstrapKey64> for CoreEngine {
    fn convert_lwe_bootstrap_key(
        &mut self,
        input: &LweBootstrapKey64,
    ) -> Result<FourierLweBootstrapKey64, LweBootstrapKeyConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_lwe_bootstrap_key_unchecked(input) })
    }

    unsafe fn convert_lwe_bootstrap_key_unchecked(
        &mut self,
        input: &LweBootstrapKey64,
    ) -> FourierLweBootstrapKey64 {
        let mut output = ImplFourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            input.glwe_dimension().to_glwe_size(),
            input.polynomial_size(),
            input.decomposition_level_count(),
            input.decomposition_base_log(),
            input.input_lwe_dimension(),
        );
        output.fill_with_forward_fourier(&input.0);
        FourierLweBootstrapKey64(output)
    }
}

impl<Key> LweBootstrapKeyConversionEngine<Key, Key> for CoreEngine
where
    Key: LweBootstrapKeyEntity + Clone,
{
    fn convert_lwe_bootstrap_key(
        &mut self,
        input: &Key,
    ) -> Result<Key, LweBootstrapKeyConversionError<Self::EngineError>> {
        Ok(unsafe { self.convert_lwe_bootstrap_key_unchecked(input) })
    }

    unsafe fn convert_lwe_bootstrap_key_unchecked(&mut self, input: &Key) -> Key {
        (*input).clone()
    }
}
