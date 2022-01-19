use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    Cleartext32, Cleartext64, LweCiphertext32, LweCiphertext64,
};
use crate::specification::engines::{
    LweCiphertextCleartextDiscardingMultiplicationEngine,
    LweCiphertextCleartextDiscardingMultiplicationError,
};

/// # Description:
/// Implementation of [`LweCiphertextCleartextDiscardingMultiplicationEngine`] for [`CoreEngine`]
/// that operates on 32 bits integers.
impl
    LweCiphertextCleartextDiscardingMultiplicationEngine<
        LweCiphertext32,
        Cleartext32,
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
    /// let input = 3_u32 << 20;
    /// let cleartext_input = 12_u32;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext: Cleartext32 = engine.create_cleartext(&cleartext_input)?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext_1 = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    /// let mut ciphertext_2 = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.discard_mul_lwe_ciphertext_cleartext(&mut ciphertext_2, &ciphertext_1, &cleartext)?;
    /// #
    /// assert_eq!(ciphertext_2.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(cleartext)?;
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext_1)?;
    /// engine.destroy(ciphertext_2)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext32,
        input_1: &LweCiphertext32,
        input_2: &Cleartext32,
    ) -> Result<(), LweCiphertextCleartextDiscardingMultiplicationError<Self::EngineError>> {
        LweCiphertextCleartextDiscardingMultiplicationError::perform_generic_checks(
            output, input_1,
        )?;
        unsafe { self.discard_mul_lwe_ciphertext_cleartext_unchecked(output, input_1, input_2) };
        Ok(())
    }

    unsafe fn discard_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input_1: &LweCiphertext32,
        input_2: &Cleartext32,
    ) {
        output.0.fill_with_scalar_mul(&input_1.0, &input_2.0);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCleartextDiscardingMultiplicationEngine`] for [`CoreEngine`]
/// that operates on 64 bits integers.
impl
    LweCiphertextCleartextDiscardingMultiplicationEngine<
        LweCiphertext64,
        Cleartext64,
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
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = 3_u64 << 50;
    /// let cleartext_input = 12_u64;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext: Cleartext64 = engine.create_cleartext(&cleartext_input)?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let ciphertext_1 = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    /// let mut ciphertext_2 = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.discard_mul_lwe_ciphertext_cleartext(&mut ciphertext_2, &ciphertext_1, &cleartext)?;
    /// #
    /// assert_eq!(ciphertext_2.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(cleartext)?;
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext_1)?;
    /// engine.destroy(ciphertext_2)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext64,
        input_1: &LweCiphertext64,
        input_2: &Cleartext64,
    ) -> Result<(), LweCiphertextCleartextDiscardingMultiplicationError<Self::EngineError>> {
        LweCiphertextCleartextDiscardingMultiplicationError::perform_generic_checks(
            output, input_1,
        )?;
        unsafe { self.discard_mul_lwe_ciphertext_cleartext_unchecked(output, input_1, input_2) };
        Ok(())
    }

    unsafe fn discard_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input_1: &LweCiphertext64,
        input_2: &Cleartext64,
    ) {
        output.0.fill_with_scalar_mul(&input_1.0, &input_2.0);
    }
}
