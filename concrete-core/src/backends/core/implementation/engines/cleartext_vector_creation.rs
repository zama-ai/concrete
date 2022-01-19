use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{CleartextVector32, CleartextVector64};
use crate::backends::core::private::crypto::encoding::CleartextList as ImplCleartextList;
use crate::specification::engines::{CleartextVectorCreationEngine, CleartextVectorCreationError};

/// # Description:
/// Implementation of [`CleartextVectorCreationEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl CleartextVectorCreationEngine<u32, CleartextVector32> for CoreEngine {
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
    /// #
    /// assert_eq!(cleartext_vector.cleartext_count(), CleartextCount(100));
    /// engine.destroy(cleartext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_cleartext_vector(
        &mut self,
        input: &[u32],
    ) -> Result<CleartextVector32, CleartextVectorCreationError<Self::EngineError>> {
        CleartextVectorCreationError::perform_generic_checks(input)?;
        Ok(unsafe { self.create_cleartext_vector_unchecked(input) })
    }

    unsafe fn create_cleartext_vector_unchecked(&mut self, input: &[u32]) -> CleartextVector32 {
        CleartextVector32(ImplCleartextList::from_container(input.to_vec()))
    }
}

/// # Description:
/// Implementation of [`CleartextVectorCreationEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl CleartextVectorCreationEngine<u64, CleartextVector64> for CoreEngine {
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
    /// #
    /// assert_eq!(cleartext_vector.cleartext_count(), CleartextCount(100));
    /// engine.destroy(cleartext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_cleartext_vector(
        &mut self,
        input: &[u64],
    ) -> Result<CleartextVector64, CleartextVectorCreationError<Self::EngineError>> {
        CleartextVectorCreationError::perform_generic_checks(input)?;
        Ok(unsafe { self.create_cleartext_vector_unchecked(input) })
    }

    unsafe fn create_cleartext_vector_unchecked(&mut self, input: &[u64]) -> CleartextVector64 {
        CleartextVector64(ImplCleartextList::from_container(input.to_vec()))
    }
}
