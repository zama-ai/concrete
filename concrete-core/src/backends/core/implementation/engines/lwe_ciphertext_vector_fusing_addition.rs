use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64,
};
use crate::specification::engines::{
    LweCiphertextVectorFusingAdditionEngine, LweCiphertextVectorFusingAdditionError,
};

/// # Description:
/// Implementation of [`LweCiphertextVectorFusingAdditionEngine`] for [`CoreEngine`]
/// that operates on 32 bits integers.
impl LweCiphertextVectorFusingAdditionEngine<LweCiphertextVector32, LweCiphertextVector32>
    for CoreEngine
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
    /// let noise = Variance::from_variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector = engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// let mut output_ciphertext_vector =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// engine.fuse_add_lwe_ciphertext_vector(&mut output_ciphertext_vector, &ciphertext_vector)?;
    /// #
    /// assert_eq!(output_ciphertext_vector.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(output_ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_add_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertextVector32,
        input: &LweCiphertextVector32,
    ) -> Result<(), LweCiphertextVectorFusingAdditionError<Self::EngineError>> {
        LweCiphertextVectorFusingAdditionError::perform_generic_checks(output, input)?;
        unsafe { self.fuse_add_lwe_ciphertext_vector_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertextVector32,
        input: &LweCiphertextVector32,
    ) {
        for (mut out, inp) in output
            .0
            .ciphertext_iter_mut()
            .zip(input.0.ciphertext_iter())
        {
            out.update_with_add(&inp);
        }
    }
}

/// # Description:
/// Implementation of [`LweCiphertextVectorFusingAdditionEngine`] for [`CoreEngine`]
/// that operates on 64 bits integers.
impl LweCiphertextVectorFusingAdditionEngine<LweCiphertextVector64, LweCiphertextVector64>
    for CoreEngine
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
    /// let noise = Variance::from_variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input_vector)?;
    /// let ciphertext_vector = engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// let mut output_ciphertext_vector =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    ///
    /// engine.fuse_add_lwe_ciphertext_vector(&mut output_ciphertext_vector, &ciphertext_vector)?;
    /// #
    /// assert_eq!(output_ciphertext_vector.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// engine.destroy(output_ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_add_lwe_ciphertext_vector(
        &mut self,
        output: &mut LweCiphertextVector64,
        input: &LweCiphertextVector64,
    ) -> Result<(), LweCiphertextVectorFusingAdditionError<Self::EngineError>> {
        LweCiphertextVectorFusingAdditionError::perform_generic_checks(output, input)?;
        unsafe { self.fuse_add_lwe_ciphertext_vector_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_add_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut LweCiphertextVector64,
        input: &LweCiphertextVector64,
    ) {
        for (mut out, inp) in output
            .0
            .ciphertext_iter_mut()
            .zip(input.0.ciphertext_iter())
        {
            out.update_with_add(&inp);
        }
    }
}
