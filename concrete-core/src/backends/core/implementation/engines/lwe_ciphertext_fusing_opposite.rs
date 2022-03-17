use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{LweCiphertext32, LweCiphertext64};
use crate::specification::engines::{
    LweCiphertextFusingOppositeEngine, LweCiphertextFusingOppositeError,
};

/// # Description:
/// Implementation of [`LweCiphertextFusingOppositeEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl LweCiphertextFusingOppositeEngine<LweCiphertext32> for CoreEngine {
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
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let mut ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.fuse_opp_lwe_ciphertext(&mut ciphertext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_opp_lwe_ciphertext(
        &mut self,
        input: &mut LweCiphertext32,
    ) -> Result<(), LweCiphertextFusingOppositeError<Self::EngineError>> {
        unsafe { self.fuse_opp_lwe_ciphertext_unchecked(input) };
        Ok(())
    }

    unsafe fn fuse_opp_lwe_ciphertext_unchecked(&mut self, input: &mut LweCiphertext32) {
        input.0.update_with_neg();
    }
}

/// # Description:
/// Implementation of [`LweCiphertextFusingOppositeEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl LweCiphertextFusingOppositeEngine<LweCiphertext64> for CoreEngine {
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
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext = engine.create_plaintext(&input)?;
    /// let mut ciphertext = engine.encrypt_lwe_ciphertext(&key, &plaintext, noise)?;
    ///
    /// engine.fuse_opp_lwe_ciphertext(&mut ciphertext)?;
    /// #
    /// assert_eq!(ciphertext.lwe_dimension(), lwe_dimension);
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn fuse_opp_lwe_ciphertext(
        &mut self,
        input: &mut LweCiphertext64,
    ) -> Result<(), LweCiphertextFusingOppositeError<Self::EngineError>> {
        unsafe { self.fuse_opp_lwe_ciphertext_unchecked(input) };
        Ok(())
    }

    unsafe fn fuse_opp_lwe_ciphertext_unchecked(&mut self, input: &mut LweCiphertext64) {
        input.0.update_with_neg();
    }
}
