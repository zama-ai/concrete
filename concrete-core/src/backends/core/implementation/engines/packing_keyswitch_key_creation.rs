use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweSecretKey32, LweSecretKey64};
use crate::backends::core::private::crypto::glwe::PackingKeyswitchKey as ImplPackingKeyswitchKey;
use crate::prelude::{
    GlweSecretKey32, GlweSecretKey64, GlweSecretKeyEntity, PackingKeyswitchKey32,
    PackingKeyswitchKey64, PackingKeyswitchKeyCreationError,
};
use crate::specification::engines::PackingKeyswitchKeyCreationEngine;
use crate::specification::entities::LweSecretKeyEntity;

/// # Description:
/// Implementation of [`PackingKeyswitchKeyCreationEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl PackingKeyswitchKeyCreationEngine<LweSecretKey32, GlweSecretKey32, PackingKeyswitchKey32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let input_lwe_dimension = LweDimension(6);
    /// let output_lwe_dimension = LweDimension(3);
    /// let decomposition_level_count = DecompositionLevelCount(2);
    /// let decomposition_base_log = DecompositionBaseLog(8);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let input_key: LweSecretKey32 = engine.create_lwe_secret_key(input_lwe_dimension)?;
    /// let output_key: LweSecretKey32 = engine.create_lwe_secret_key(output_lwe_dimension)?;
    ///
    /// let packing_keyswitch_key = engine.create_lwe_keyswitch_key(
    ///     &input_key,
    ///     &output_key,
    ///     decomposition_level_count,
    ///     decomposition_base_log,
    ///     noise,
    /// )?;
    /// #
    /// assert_eq!(
    /// #     packing_keyswitch_key.decomposition_level_count(),
    /// #     decomposition_level_count
    /// # );
    /// assert_eq!(
    /// #     packing_keyswitch_key.decomposition_base_log(),
    /// #     decomposition_base_log
    /// # );
    /// assert_eq!(packing_keyswitch_key.input_lwe_dimension(), input_lwe_dimension);
    /// assert_eq!(packing_keyswitch_key.output_lwe_dimension(), output_lwe_dimension);
    ///
    /// engine.destroy(input_key)?;
    /// engine.destroy(output_key)?;
    /// engine.destroy(packing_keyswitch_key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_packing_keyswitch_key(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Result<PackingKeyswitchKey32, PackingKeyswitchKeyCreationError<Self::EngineError>> {
        PackingKeyswitchKeyCreationError::perform_generic_checks(
            decomposition_level_count,
            decomposition_base_log,
            32,
        )?;
        Ok(unsafe {
            self.create_packing_keyswitch_key_unchecked(
                input_key,
                output_key,
                decomposition_level_count,
                decomposition_base_log,
                noise,
            )
        })
    }

    unsafe fn create_packing_keyswitch_key_unchecked(
        &mut self,
        input_key: &LweSecretKey32,
        output_key: &GlweSecretKey32,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> PackingKeyswitchKey32 {
        let mut ksk = ImplPackingKeyswitchKey::allocate(
            0,
            decomposition_level_count,
            decomposition_base_log,
            input_key.lwe_dimension(),
            output_key.glwe_dimension(),
            output_key.polynomial_size(),
        );
        ksk.fill_with_packing_keyswitch_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        PackingKeyswitchKey32(ksk)
    }
}

/// # Description:
/// Implementation of [`PackingKeyswitchKeyCreationEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl PackingKeyswitchKeyCreationEngine<LweSecretKey64, GlweSecretKey64, PackingKeyswitchKey64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    /// };
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let input_lwe_dimension = LweDimension(6);
    /// let output_lwe_dimension = LweDimension(3);
    /// let decomposition_level_count = DecompositionLevelCount(2);
    /// let decomposition_base_log = DecompositionBaseLog(8);
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let input_key: LweSecretKey64 = engine.create_lwe_secret_key(input_lwe_dimension)?;
    /// let output_key: LweSecretKey64 = engine.create_lwe_secret_key(output_lwe_dimension)?;
    ///
    /// let packing_keyswitch_key = engine.create_lwe_keyswitch_key(
    ///     &input_key,
    ///     &output_key,
    ///     decomposition_level_count,
    ///     decomposition_base_log,
    ///     noise,
    /// )?;
    /// #
    /// assert_eq!(
    /// #     packing_keyswitch_key.decomposition_level_count(),
    /// #     decomposition_level_count
    /// # );
    /// assert_eq!(
    /// #     packing_keyswitch_key.decomposition_base_log(),
    /// #     decomposition_base_log
    /// # );
    /// assert_eq!(packing_keyswitch_key.input_lwe_dimension(), input_lwe_dimension);
    /// assert_eq!(packing_keyswitch_key.output_lwe_dimension(), output_lwe_dimension);
    ///
    /// engine.destroy(input_key)?;
    /// engine.destroy(output_key)?;
    /// engine.destroy(packing_keyswitch_key)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_packing_keyswitch_key(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Result<PackingKeyswitchKey64, PackingKeyswitchKeyCreationError<Self::EngineError>> {
        PackingKeyswitchKeyCreationError::perform_generic_checks(
            decomposition_level_count,
            decomposition_base_log,
            64,
        )?;
        Ok(unsafe {
            self.create_packing_keyswitch_key_unchecked(
                input_key,
                output_key,
                decomposition_level_count,
                decomposition_base_log,
                noise,
            )
        })
    }

    unsafe fn create_packing_keyswitch_key_unchecked(
        &mut self,
        input_key: &LweSecretKey64,
        output_key: &GlweSecretKey64,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> PackingKeyswitchKey64 {
        let mut ksk = ImplPackingKeyswitchKey::allocate(
            0,
            decomposition_level_count,
            decomposition_base_log,
            input_key.lwe_dimension(),
            output_key.glwe_dimension(),
            output_key.polynomial_size(),
        );
        ksk.fill_with_packing_keyswitch_key(
            &input_key.0,
            &output_key.0,
            noise,
            &mut self.encryption_generator,
        );
        PackingKeyswitchKey64(ksk)
    }
}
