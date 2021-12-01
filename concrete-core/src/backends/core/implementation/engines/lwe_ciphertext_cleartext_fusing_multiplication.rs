use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    Cleartext32, Cleartext64, LweCiphertext32, LweCiphertext64,
};
use crate::specification::engines::{
    LweCiphertextCleartextFusingMultiplicationEngine,
    LweCiphertextCleartextFusingMultiplicationError,
};

/// # Description:
/// Implementation of [`LweCiphertextCleartextFusingMultiplicationEngine`] for [`CoreEngine`] that
/// operates on 32 bits integers.
impl LweCiphertextCleartextFusingMultiplicationEngine<LweCiphertext32, Cleartext32> for CoreEngine {
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
    /// let mut ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.fuse_mul_lwe_ciphertext_cleartext(&mut ciphertext, &cleartext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Cleartext32,
    ) -> Result<(), LweCiphertextCleartextFusingMultiplicationError<Self::EngineError>> {
        unsafe { self.fuse_mul_lwe_ciphertext_cleartext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext32,
        input: &Cleartext32,
    ) {
        output.0.update_with_scalar_mul(input.0);
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCleartextFusingMultiplicationEngine`] for [`CoreEngine`] that
/// operates on 64 bits integers.
impl LweCiphertextCleartextFusingMultiplicationEngine<LweCiphertext64, Cleartext64> for CoreEngine {
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
    /// let mut ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.fuse_mul_lwe_ciphertext_cleartext(&mut ciphertext, &cleartext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_mul_lwe_ciphertext_cleartext(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Cleartext64,
    ) -> Result<(), LweCiphertextCleartextFusingMultiplicationError<Self::EngineError>> {
        unsafe { self.fuse_mul_lwe_ciphertext_cleartext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn fuse_mul_lwe_ciphertext_cleartext_unchecked(
        &mut self,
        output: &mut LweCiphertext64,
        input: &Cleartext64,
    ) {
        output.0.update_with_scalar_mul(input.0);
    }
}
