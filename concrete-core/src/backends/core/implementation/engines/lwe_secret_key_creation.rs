use concrete_commons::parameters::LweDimension;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweSecretKey32, LweSecretKey64};
use crate::backends::core::private::crypto::secret::LweSecretKey as ImplLweSecretKey;
use crate::specification::engines::{LweSecretKeyCreationEngine, LweSecretKeyCreationError};

impl LweSecretKeyCreationEngine<LweSecretKey32> for CoreEngine {
    fn create_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Result<LweSecretKey32, LweSecretKeyCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_lwe_secret_key_unchecked(lwe_dimension) })
    }

    unsafe fn create_lwe_secret_key_unchecked(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> LweSecretKey32 {
        LweSecretKey32(ImplLweSecretKey::generate_binary(
            lwe_dimension,
            &mut self.secret_generator,
        ))
    }
}

impl LweSecretKeyCreationEngine<LweSecretKey64> for CoreEngine {
    fn create_lwe_secret_key(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> Result<LweSecretKey64, LweSecretKeyCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_lwe_secret_key_unchecked(lwe_dimension) })
    }

    unsafe fn create_lwe_secret_key_unchecked(
        &mut self,
        lwe_dimension: LweDimension,
    ) -> LweSecretKey64 {
        LweSecretKey64(ImplLweSecretKey::generate_binary(
            lwe_dimension,
            &mut self.secret_generator,
        ))
    }
}
