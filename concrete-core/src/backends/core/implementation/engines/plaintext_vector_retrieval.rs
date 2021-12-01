use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{PlaintextVector32, PlaintextVector64};
use crate::backends::core::private::math::tensor::AsRefTensor;
use crate::specification::engines::{
    PlaintextVectorRetrievalEngine, PlaintextVectorRetrievalError,
};

/// # Description:
/// Implementation of [`PlaintextVectorRetrievalEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl PlaintextVectorRetrievalEngine<PlaintextVector32, u32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::PlaintextCount;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u32 << 20; 3];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector32 = engine.create_plaintext_vector(&input)?;
    /// let output: Vec<u32> = engine.retrieve_plaintext_vector(&plaintext_vector)?;
    /// #
    /// assert_eq!(output[0], 3_u32 << 20);
    /// engine.destroy(plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_plaintext_vector(
        &mut self,
        plaintext: &PlaintextVector32,
    ) -> Result<Vec<u32>, PlaintextVectorRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_plaintext_vector_unchecked(plaintext) })
    }

    unsafe fn retrieve_plaintext_vector_unchecked(
        &mut self,
        plaintext: &PlaintextVector32,
    ) -> Vec<u32> {
        plaintext.0.as_tensor().as_container().to_vec()
    }
}

/// # Description:
/// Implementation of [`PlaintextVectorRetrievalEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl PlaintextVectorRetrievalEngine<PlaintextVector64, u64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::PlaintextCount;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = vec![3_u64 << 20; 3];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// let output: Vec<u64> = engine.retrieve_plaintext_vector(&plaintext_vector)?;
    /// #
    /// assert_eq!(output[0], 3_u64 << 20);
    /// engine.destroy(plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_plaintext_vector(
        &mut self,
        plaintext: &PlaintextVector64,
    ) -> Result<Vec<u64>, PlaintextVectorRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_plaintext_vector_unchecked(plaintext) })
    }

    unsafe fn retrieve_plaintext_vector_unchecked(
        &mut self,
        plaintext: &PlaintextVector64,
    ) -> Vec<u64> {
        plaintext.0.as_tensor().as_container().to_vec()
    }
}
