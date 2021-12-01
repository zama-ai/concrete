use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::CiphertextCount;

use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertextVector32, LweCiphertextVector64, LweSecretKey32, LweSecretKey64,
    PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::lwe::LweList as ImplLweList;
use crate::specification::engines::{
    LweCiphertextVectorEncryptionEngine, LweCiphertextVectorEncryptionError,
};
use crate::specification::entities::{LweSecretKeyEntity, PlaintextVectorEntity};

/// # Description:
/// Implementation of [`LweCiphertextVectorEncryptionEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl LweCiphertextVectorEncryptionEngine<LweSecretKey32, PlaintextVector32, LweCiphertextVector32>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(6);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u32 << 20; 3];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey32 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    ///
    /// let mut ciphertext_vector: LweCiphertextVector32 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// #
    /// assert_eq!(ciphertext_vector.lwe_dimension(), lwe_dimension);
    /// assert_eq!(
    /// #    ciphertext_vector.lwe_ciphertext_count(),
    /// #    LweCiphertextCount(3)
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> Result<LweCiphertextVector32, LweCiphertextVectorEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey32,
        input: &PlaintextVector32,
        noise: Variance,
    ) -> LweCiphertextVector32 {
        let mut vector = ImplLweList::allocate(
            0u32,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(input.plaintext_count().0),
        );
        key.0
            .encrypt_lwe_list(&mut vector, &input.0, noise, &mut self.encryption_generator);
        LweCiphertextVector32(vector)
    }
}

/// # Description:
/// Implementation of [`LweCiphertextVectorEncryptionEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl LweCiphertextVectorEncryptionEngine<LweSecretKey64, PlaintextVector64, LweCiphertextVector64>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let lwe_dimension = LweDimension(6);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = vec![3_u64 << 50; 3];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let key: LweSecretKey64 = engine.create_lwe_secret_key(lwe_dimension)?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    ///
    /// let mut ciphertext_vector: LweCiphertextVector64 =
    ///     engine.encrypt_lwe_ciphertext_vector(&key, &plaintext_vector, noise)?;
    /// #
    /// assert_eq!(ciphertext_vector.lwe_dimension(), lwe_dimension);
    /// assert_eq!(
    /// #     ciphertext_vector.lwe_ciphertext_count(),
    /// #     LweCiphertextCount(3)
    /// # );
    ///
    /// engine.destroy(key)?;
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn encrypt_lwe_ciphertext_vector(
        &mut self,
        key: &LweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> Result<LweCiphertextVector64, LweCiphertextVectorEncryptionError<Self::EngineError>> {
        Ok(unsafe { self.encrypt_lwe_ciphertext_vector_unchecked(key, input, noise) })
    }

    unsafe fn encrypt_lwe_ciphertext_vector_unchecked(
        &mut self,
        key: &LweSecretKey64,
        input: &PlaintextVector64,
        noise: Variance,
    ) -> LweCiphertextVector64 {
        let mut vector = ImplLweList::allocate(
            0u64,
            key.lwe_dimension().to_lwe_size(),
            CiphertextCount(input.plaintext_count().0),
        );
        key.0
            .encrypt_lwe_list(&mut vector, &input.0, noise, &mut self.encryption_generator);
        LweCiphertextVector64(vector)
    }
}
