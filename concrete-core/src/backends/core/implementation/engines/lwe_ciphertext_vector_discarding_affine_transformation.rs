use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    CleartextVector32, CleartextVector64, LweCiphertext32, LweCiphertext64, LweCiphertextVector32,
    LweCiphertextVector64, Plaintext32, Plaintext64,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingAffineTransformationEngine,
    LweCiphertextVectorDiscardingAffineTransformationError,
};
use crate::specification::entities::{
    CleartextVectorEntity, LweCiphertextEntity, LweCiphertextVectorEntity,
};

/// # Description:
/// Implementation of [`LweCiphertextVectorDiscardingAffineTransformationEngine`] for [`CoreEngine`]
/// that operates on 32 bits integers.
impl
    LweCiphertextVectorDiscardingAffineTransformationEngine<
        LweCiphertextVector32,
        CleartextVector32,
        Plaintext32,
        LweCiphertext32,
    > for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(2);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input_vector = vec![3_u32 << 20; 8];
    /// let weights_input = vec![2_u32; 8];
    /// let bias_input = 8_u32 << 20;
    /// let noise = Variance::from_variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let weights: CleartextVector32 = engine.create_cleartext_vector(&input_vector)?;
    /// let bias: Plaintext32 = engine.create_plaintext(&bias_input)?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector = engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// let mut output_ciphertext = engine.zero_encrypt_lwe_ciphertext(&key, noise)?;
    ///
    /// engine.discard_affine_transform_lwe_ciphertext_vector(
    ///     &mut output_ciphertext,
    ///     &ciphertext_vector,
    ///     &weights,
    ///     &bias,
    /// )?;
    /// #
    /// assert_eq!(output_ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(weights)?;
    /// engine.destroy(bias)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(output_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_affine_transform_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertext32,
        inputs: &LweCiphertextVector32,
        weights: &CleartextVector32,
        bias: &Plaintext32,
    ) -> Result<(), LweCiphertextVectorDiscardingAffineTransformationError<Self::EngineError>> {
        if output.lwe_dimension() != inputs.lwe_dimension() {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::LweDimensionMismatch,
            );
        }
        if inputs.lwe_ciphertext_count().0 != weights.cleartext_count().0 {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::CleartextCountMismatch,
            );
        }
        unsafe {
            self.discard_affine_transform_lwe_ciphertext_vector_unchecked(
                output, inputs, weights, bias,
            )
        };
        Ok(())
    }

    unsafe fn discard_affine_transform_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        inputs: &LweCiphertextVector32,
        weights: &CleartextVector32,
        bias: &Plaintext32,
    ) {
        output
            .0
            .fill_with_multisum_with_bias(&inputs.0, &weights.0, &bias.0);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextVectorDiscardingAffineTransformationEngine`] for [`CoreEngine`]
/// that operates on 64 bits integers.
impl
    LweCiphertextVectorDiscardingAffineTransformationEngine<
        LweCiphertextVector64,
        CleartextVector64,
        Plaintext64,
        LweCiphertext64,
    > for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(2);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input_vector = vec![3_u64 << 50; 8];
    /// let weights_input = vec![2_u64; 8];
    /// let bias_input = 8_u64 << 50;
    /// let noise = Variance::from_variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let weights: CleartextVector64 = engine.create_cleartext_vector(&input_vector)?;
    /// let bias: Plaintext64 = engine.create_plaintext(&bias_input)?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector = engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// let mut output_ciphertext = engine.zero_encrypt_lwe_ciphertext(&key, noise)?;
    ///
    /// engine.discard_affine_transform_lwe_ciphertext_vector(
    ///     &mut output_ciphertext,
    ///     &ciphertext_vector,
    ///     &weights,
    ///     &bias,
    /// )?;
    /// #
    /// assert_eq!(output_ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(weights)?;
    /// engine.destroy(bias)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(output_ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_affine_transform_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertext64,
        inputs: &LweCiphertextVector64,
        weights: &CleartextVector64,
        bias: &Plaintext64,
    ) -> Result<(), LweCiphertextVectorDiscardingAffineTransformationError<Self::EngineError>> {
        if output.lwe_dimension() != inputs.lwe_dimension() {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::LweDimensionMismatch,
            );
        }
        if inputs.lwe_ciphertext_count().0 != weights.cleartext_count().0 {
            return Err(
                LweCiphertextVectorDiscardingAffineTransformationError::CleartextCountMismatch,
            );
        }
        unsafe {
            self.discard_affine_transform_lwe_ciphertext_vector_unchecked(
                output, inputs, weights, bias,
            )
        };
        Ok(())
    }

    unsafe fn discard_affine_transform_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        inputs: &LweCiphertextVector64,
        weights: &CleartextVector64,
        bias: &Plaintext64,
    ) {
        output
            .0
            .fill_with_multisum_with_bias(&inputs.0, &weights.0, &bias.0);
    }
}
