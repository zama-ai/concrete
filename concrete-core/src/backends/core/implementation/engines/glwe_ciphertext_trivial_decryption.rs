use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::backends::core::private::math::tensor::AsRefTensor;
use crate::prelude::{
    CoreEngine, GlweCiphertext32, GlweCiphertext64, GlweCiphertextTrivialDecryptionEngine,
    GlweCiphertextTrivialDecryptionError, PlaintextVector32, PlaintextVector64,
};

impl GlweCiphertextTrivialDecryptionEngine<GlweCiphertext32, PlaintextVector32> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let input = vec![3_u32 << 20; polynomial_size.0];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext: GlweCiphertext32 = engine
    ///     .trivially_encrypt_glwe_ciphertext(glwe_dimension.to_glwe_size(), &plaintext_vector)?;
    /// let output: PlaintextVector32 = engine.trivially_decrypt_glwe_ciphertext(&ciphertext)?;
    ///
    /// assert_eq!(output.plaintext_count(), PlaintextCount(polynomial_size.0));
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(output)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        input: &GlweCiphertext32,
    ) -> Result<PlaintextVector32, GlweCiphertextTrivialDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.trivially_decrypt_glwe_ciphertext_unchecked(input) })
    }

    unsafe fn trivially_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        input: &GlweCiphertext32,
    ) -> PlaintextVector32 {
        PlaintextVector32(ImplPlaintextList::from_container(
            input.0.get_body().as_tensor().as_container().to_vec(),
        ))
    }
}

impl GlweCiphertextTrivialDecryptionEngine<GlweCiphertext64, PlaintextVector64> for CoreEngine {
    /// # Example:
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::prelude::*;
    ///
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(4);
    /// let input = vec![3_u64 << 20; polynomial_size.0];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// let ciphertext: GlweCiphertext64 = engine
    ///     .trivially_encrypt_glwe_ciphertext(glwe_dimension.to_glwe_size(), &plaintext_vector)?;
    /// let output: PlaintextVector64 = engine.trivially_decrypt_glwe_ciphertext(&ciphertext)?;
    ///
    /// assert_eq!(output.plaintext_count(), PlaintextCount(polynomial_size.0));
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    /// engine.destroy(output)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_decrypt_glwe_ciphertext(
        &mut self,
        input: &GlweCiphertext64,
    ) -> Result<PlaintextVector64, GlweCiphertextTrivialDecryptionError<Self::EngineError>> {
        Ok(unsafe { self.trivially_decrypt_glwe_ciphertext_unchecked(input) })
    }

    unsafe fn trivially_decrypt_glwe_ciphertext_unchecked(
        &mut self,
        input: &GlweCiphertext64,
    ) -> PlaintextVector64 {
        PlaintextVector64(ImplPlaintextList::from_container(
            input.0.get_body().as_tensor().as_container().to_vec(),
        ))
    }
}
