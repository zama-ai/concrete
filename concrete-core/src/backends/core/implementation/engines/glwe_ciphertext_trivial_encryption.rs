use concrete_commons::parameters::GlweSize;

use crate::backends::core::entities::{
    GlweCiphertext32, GlweCiphertext64, PlaintextVector32, PlaintextVector64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::engines::{
    GlweCiphertextTrivialEncryptionEngine, GlweCiphertextTrivialEncryptionError,
};

use crate::backends::core::engines::CoreEngine;

impl GlweCiphertextTrivialEncryptionEngine<PlaintextVector32, GlweCiphertext32> for CoreEngine {
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
    ///
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_glwe_ciphertext(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector32,
    ) -> Result<GlweCiphertext32, GlweCiphertextTrivialEncryptionError<Self::EngineError>> {
        unsafe { Ok(self.trivially_encrypt_glwe_ciphertext_unchecked(glwe_size, input)) }
    }

    unsafe fn trivially_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector32,
    ) -> GlweCiphertext32 {
        let ciphertext: ImplGlweCiphertext<Vec<u32>> =
            ImplGlweCiphertext::new_trivial_encryption(glwe_size, &input.0);
        GlweCiphertext32(ciphertext)
    }
}

impl GlweCiphertextTrivialEncryptionEngine<PlaintextVector64, GlweCiphertext64> for CoreEngine {
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
    ///
    /// assert_eq!(ciphertext.glwe_dimension(), glwe_dimension);
    /// assert_eq!(ciphertext.polynomial_size(), polynomial_size);
    ///
    /// engine.destroy(plaintext_vector)?;
    /// engine.destroy(ciphertext)?;
    ///
    /// # Ok(())
    /// # }
    /// ```
    fn trivially_encrypt_glwe_ciphertext(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector64,
    ) -> Result<GlweCiphertext64, GlweCiphertextTrivialEncryptionError<Self::EngineError>> {
        unsafe { Ok(self.trivially_encrypt_glwe_ciphertext_unchecked(glwe_size, input)) }
    }

    unsafe fn trivially_encrypt_glwe_ciphertext_unchecked(
        &mut self,
        glwe_size: GlweSize,
        input: &PlaintextVector64,
    ) -> GlweCiphertext64 {
        let ciphertext: ImplGlweCiphertext<Vec<u64>> =
            ImplGlweCiphertext::new_trivial_encryption(glwe_size, &input.0);
        GlweCiphertext64(ciphertext)
    }
}
