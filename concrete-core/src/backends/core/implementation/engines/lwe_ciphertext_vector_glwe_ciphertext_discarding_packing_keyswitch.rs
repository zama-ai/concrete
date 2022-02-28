use crate::backends::core::implementation::engines::CoreEngine;
use crate::prelude::{
    GlweCiphertext32, GlweCiphertext64, LweCiphertextVector32, LweCiphertextVector64,
    PackingKeyswitchKey32, PackingKeyswitchKey64,
};
use crate::specification::engines::{
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine,
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError,
};

/// # Description:
/// Implementation of [`LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine`] for
/// [`CoreEngine`] that operates on 32 bits integers.
impl
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine<
        PackingKeyswitchKey32,
        LweCiphertextVector32,
        GlweCiphertext32,
    > for CoreEngine
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
    /// let output_glwe_dimension = GlweDimension(3);
    /// let decomposition_level_count = DecompositionLevelCount(2);
    /// let decomposition_base_log = DecompositionBaseLog(8);
    /// let polynomial_size = PolynomialSize(256);
    /// let noise = Variance(2_f64.powf(-25.));
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input_vector = vec![3_u32 << 20, 256];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let input_key: LweSecretKey32 = engine.create_lwe_secret_key(input_lwe_dimension)?;
    /// let output_key: GlweSecretKey32 =
    ///     engine.create_glwe_secret_key(output_glwe_dimension, polynomial_size)?;
    /// let packing_keyswitch_key = engine.create_packing_keyswitch_key(
    ///     &input_key,
    ///     &output_key,
    ///     decomposition_level_count,
    ///     decomposition_base_log,
    ///     noise,
    /// )?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector =
    ///     engine.encrypt_lwe_ciphertext_vector(&input_key, &plaintext_vector, noise)?;
    /// let mut ciphertext_output = engine.zero_encrypt_glwe_ciphertext(&output_key, noise)?;
    ///
    /// engine.discard_packing_keyswitch_lwe_ciphertext_vector(
    ///     &mut ciphertext_output,
    ///     &ciphertext_vector,
    ///     &packing_keyswitch_key,
    /// )?;
    /// #
    /// assert_eq!(ciphertext_output.glwe_dimension(), output_glwe_dimension);
    ///
    /// engine.destroy(input_key)?;
    /// engine.destroy(output_key)?;
    /// engine.destroy(packing_keyswitch_key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(ciphertext_output)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_packing_keyswitch_lwe_ciphertext_vector(
        &mut self,
        output: &mut GlweCiphertext32,
        input: &LweCiphertextVector32,
        ksk: &PackingKeyswitchKey32,
    ) -> Result<
        (),
        LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError<Self::EngineError>,
    > {
        LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError::perform_generic_checks(
            output, input, ksk,
        )?;
        unsafe {
            self.discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(output, input, ksk)
        };
        Ok(())
    }

    unsafe fn discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut GlweCiphertext32,
        input: &LweCiphertextVector32,
        ksk: &PackingKeyswitchKey32,
    ) {
        ksk.0.packing_keyswitch(&mut output.0, &input.0);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine`] for
/// [`CoreEngine`] that operates on 64 bits integers.
impl
    LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchEngine<
        PackingKeyswitchKey64,
        LweCiphertextVector64,
        GlweCiphertext64,
    > for CoreEngine
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
    /// let output_glwe_dimension = GlweDimension(3);
    /// let decomposition_level_count = DecompositionLevelCount(2);
    /// let decomposition_base_log = DecompositionBaseLog(8);
    /// let polynomial_size = PolynomialSize(256);
    /// let noise = Variance(2_f64.powf(-25.));
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input_vector = vec![3_u64 << 50, 256];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let input_key: LweSecretKey64 = engine.create_lwe_secret_key(input_lwe_dimension)?;
    /// let output_key: GlweSecretKey64 =
    ///     engine.create_glwe_secret_key(output_glwe_dimension, polynomial_size)?;
    /// let packing_keyswitch_key = engine.create_packing_keyswitch_key(
    ///     &input_key,
    ///     &output_key,
    ///     decomposition_level_count,
    ///     decomposition_base_log,
    ///     noise,
    /// )?;
    /// let plaintext_vector = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector =
    ///     engine.encrypt_lwe_ciphertext_vector(&input_key, &plaintext_vector, noise)?;
    /// let mut ciphertext_output = engine.zero_encrypt_glwe_ciphertext(&output_key, noise)?;
    ///
    /// engine.discard_packing_keyswitch_lwe_ciphertext_vector(
    ///     &mut ciphertext_output,
    ///     &ciphertext_vector,
    ///     &packing_keyswitch_key,
    /// )?;
    /// #
    /// assert_eq!(ciphertext_output.glwe_dimension(), output_glwe_dimension);
    ///
    /// engine.destroy(input_key)?;
    /// engine.destroy(output_key)?;
    /// engine.destroy(packing_keyswitch_key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(ciphertext_output)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_packing_keyswitch_lwe_ciphertext_vector(
        &mut self,
        output: &mut GlweCiphertext64,
        input: &LweCiphertextVector64,
        ksk: &PackingKeyswitchKey64,
    ) -> Result<
        (),
        LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError<Self::EngineError>,
    > {
        LweCiphertextVectorGlweCiphertextDiscardingPackingKeyswitchError::perform_generic_checks(
            output, input, ksk,
        )?;
        unsafe {
            self.discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(output, input, ksk)
        };
        Ok(())
    }

    unsafe fn discard_packing_keyswitch_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut GlweCiphertext64,
        input: &LweCiphertextVector64,
        ksk: &PackingKeyswitchKey64,
    ) {
        ksk.0.packing_keyswitch(&mut output.0, &input.0);
    }
}
