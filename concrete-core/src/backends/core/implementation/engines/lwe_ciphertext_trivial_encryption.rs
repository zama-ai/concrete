use crate::prelude::{CoreEngine, LweCiphertext32, LweCiphertext64, Plaintext32, Plaintext64};
use crate::specification::engines::{
    LweCiphertextTrivialEncryptionEngine, LweCiphertextTrivialEncryptionError,
};
use concrete_commons::parameters::LweSize;

use crate::backends::core::private::crypto::lwe::LweCiphertext as ImplLweCiphertext;

impl LweCiphertextTrivialEncryptionEngine<Plaintext32, LweCiphertext32> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_size = LweSize(10);
    /// let input = 3_u32 << 20;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext32 = engine.create_plaintext(&input)?;
    /// // DISCLAIMER: trivial encryption is NOT secure, and DOES NOT hide the message at all.
    /// let ciphertext: LweCiphertext32 =
    ///     engine.trivially_encrypt_lwe_ciphertext(lwe_size, &plaintext)?;
    ///
    /// assert_eq!(ciphertext.lwe_dimension().to_lwe_size(), lwe_size);
    ///
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_lwe_ciphertext(
        &mut self,
        lwe_size: LweSize,
        input: &Plaintext32,
    ) -> Result<LweCiphertext32, LweCiphertextTrivialEncryptionError<Self::EngineError>> {
        unsafe { Ok(self.trivially_encrypt_lwe_ciphertext_unchecked(lwe_size, input)) }
    }

    unsafe fn trivially_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        lwe_size: LweSize,
        input: &Plaintext32,
    ) -> LweCiphertext32 {
        let ciphertext = ImplLweCiphertext::new_trivial_encryption(lwe_size, &input.0);
        LweCiphertext32(ciphertext)
    }
}

impl LweCiphertextTrivialEncryptionEngine<Plaintext64, LweCiphertext64> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{CiphertextCount, LweSize};
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_size = LweSize(10);
    /// let input = 3_u64 << 20;
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext64 = engine.create_plaintext(&input)?;
    /// // DISCLAIMER: trivial encryption is NOT secure, and DOES NOT hide the message at all.
    /// let ciphertext: LweCiphertext64 =
    ///     engine.trivially_encrypt_lwe_ciphertext(lwe_size, &plaintext)?;
    ///
    /// assert_eq!(ciphertext.lwe_dimension().to_lwe_size(), lwe_size);
    ///
    /// engine.destroy(plaintext)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_lwe_ciphertext(
        &mut self,
        lwe_size: LweSize,
        input: &Plaintext64,
    ) -> Result<LweCiphertext64, LweCiphertextTrivialEncryptionError<Self::EngineError>> {
        unsafe { Ok(self.trivially_encrypt_lwe_ciphertext_unchecked(lwe_size, input)) }
    }

    unsafe fn trivially_encrypt_lwe_ciphertext_unchecked(
        &mut self,
        lwe_size: LweSize,
        input: &Plaintext64,
    ) -> LweCiphertext64 {
        let ciphertext = ImplLweCiphertext::new_trivial_encryption(lwe_size, &input.0);
        LweCiphertext64(ciphertext)
    }
}
