use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{CleartextVector32, CleartextVector64};
use crate::backends::core::private::math::tensor::AsRefTensor;
use crate::specification::engines::{
    CleartextVectorRetrievalEngine, CleartextVectorRetrievalError,
};

/// # Description:
/// Implementation of [`CleartextVectorRetrievalEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl CleartextVectorRetrievalEngine<CleartextVector32, u32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::CleartextCount;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let input = vec![3_u32; 100];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext_vector: CleartextVector32 = engine.create_cleartext_vector(&input)?;
    /// let retrieved: Vec<u32> = engine.retrieve_cleartext_vector(&cleartext_vector)?;
    ///
    /// assert_eq!(retrieved[0], 3_u32);
    /// engine.destroy(cleartext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_cleartext_vector(
        &mut self,
        cleartext: &CleartextVector32,
    ) -> Result<Vec<u32>, CleartextVectorRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_cleartext_vector_unchecked(cleartext) })
    }

    unsafe fn retrieve_cleartext_vector_unchecked(
        &mut self,
        cleartext: &CleartextVector32,
    ) -> Vec<u32> {
        cleartext.0.as_tensor().as_container().to_vec()
    }
}

/// # Description:
/// Implementation of [`CleartextVectorRetrievalEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl CleartextVectorRetrievalEngine<CleartextVector64, u64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::CleartextCount;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let input = vec![3_u64; 100];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext_vector: CleartextVector64 = engine.create_cleartext_vector(&input)?;
    /// let retrieved: Vec<u64> = engine.retrieve_cleartext_vector(&cleartext_vector)?;
    ///
    /// assert_eq!(retrieved[0], 3_u64);
    /// engine.destroy(cleartext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_cleartext_vector(
        &mut self,
        cleartext: &CleartextVector64,
    ) -> Result<Vec<u64>, CleartextVectorRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_cleartext_vector_unchecked(cleartext) })
    }

    unsafe fn retrieve_cleartext_vector_unchecked(
        &mut self,
        cleartext: &CleartextVector64,
    ) -> Vec<u64> {
        cleartext.0.as_tensor().as_container().to_vec()
    }
}
