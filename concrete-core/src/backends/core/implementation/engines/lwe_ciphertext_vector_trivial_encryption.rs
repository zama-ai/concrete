use crate::backends::core::private::crypto::lwe::LweList as ImplLweList;
use crate::prelude::{
    CoreEngine, LweCiphertextVector32, LweCiphertextVector64,
    LweCiphertextVectorTrivialEncryptionEngine, LweCiphertextVectorTrivialEncryptionError,
    PlaintextVector32, PlaintextVector64,
};
use concrete_commons::parameters::LweSize;

impl LweCiphertextVectorTrivialEncryptionEngine<PlaintextVector32, LweCiphertextVector32>
    for CoreEngine
{
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
    /// let input = vec![3_u32 << 20; 3];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector32 =
    ///     engine.trivially_encrypt_lwe_ciphertext_vector(lwe_size, &plaintext_vector)?;
    ///
    /// assert_eq!(ciphertext_vector.lwe_dimension().to_lwe_size(), lwe_size);
    /// assert_eq!(
    ///     ciphertext_vector.lwe_ciphertext_count().0,
    ///     plaintext_vector.plaintext_count().0
    /// );
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_lwe_ciphertext_vector(
        &mut self,
        lwe_size: LweSize,
        input: &PlaintextVector32,
    ) -> Result<LweCiphertextVector32, LweCiphertextVectorTrivialEncryptionError<Self::EngineError>>
    {
        unsafe { Ok(self.trivially_encrypt_lwe_ciphertext_vector_unchecked(lwe_size, input)) }
    }

    unsafe fn trivially_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        lwe_size: LweSize,
        input: &PlaintextVector32,
    ) -> LweCiphertextVector32 {
        let ciphertexts = ImplLweList::new_trivial_encryption(lwe_size, &input.0);

        LweCiphertextVector32(ciphertexts)
    }
}

impl LweCiphertextVectorTrivialEncryptionEngine<PlaintextVector64, LweCiphertextVector64>
    for CoreEngine
{
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
    /// let input = vec![3_u64 << 20; 3];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext_vector: LweCiphertextVector64 =
    ///     engine.trivially_encrypt_lwe_ciphertext_vector(lwe_size, &plaintext_vector)?;
    ///
    /// assert_eq!(ciphertext_vector.lwe_dimension().to_lwe_size(), lwe_size);
    /// assert_eq!(
    ///     ciphertext_vector.lwe_ciphertext_count().0,
    ///     plaintext_vector.plaintext_count().0
    /// );
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_lwe_ciphertext_vector(
        &mut self,
        lwe_size: LweSize,
        input: &PlaintextVector64,
    ) -> Result<LweCiphertextVector64, LweCiphertextVectorTrivialEncryptionError<Self::EngineError>>
    {
        unsafe { Ok(self.trivially_encrypt_lwe_ciphertext_vector_unchecked(lwe_size, input)) }
    }

    unsafe fn trivially_encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        lwe_size: LweSize,
        input: &PlaintextVector64,
    ) -> LweCiphertextVector64 {
        let ciphertexts = ImplLweList::new_trivial_encryption(lwe_size, &input.0);

        LweCiphertextVector64(ciphertexts)
    }
}
