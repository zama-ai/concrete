use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{PlaintextVector32, PlaintextVector64};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{PlaintextVectorCreationEngine, PlaintextVectorCreationError};

/// # Description:
/// Implementation of [`PlaintextVectorCreationEngine`] for [`CoreEngine`] that operates on
/// 32 bits integers.
impl PlaintextVectorCreationEngine<u32, PlaintextVector32> for CoreEngine {
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
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(3));
    /// engine.destroy(plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_plaintext_vector(
        &mut self,
        input: &[u32],
    ) -> Result<PlaintextVector32, PlaintextVectorCreationError<Self::EngineError>> {
        if input.is_empty() {
            return Err(PlaintextVectorCreationError::EmptyInput);
        }
        Ok(unsafe { self.create_plaintext_vector_unchecked(input) })
    }

    unsafe fn create_plaintext_vector_unchecked(&mut self, input: &[u32]) -> PlaintextVector32 {
        PlaintextVector32(ImplPlaintextList::from_container(input.to_vec()))
    }
}

/// # Description:
/// Implementation of [`PlaintextVectorCreationEngine`] for [`CoreEngine`] that operates on
/// 64 bits integers.
impl PlaintextVectorCreationEngine<u64, PlaintextVector64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::PlaintextCount;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input = vec![3_u64 << 50; 3];
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext_vector: PlaintextVector64 = engine.create_plaintext_vector(&input)?;
    /// #
    /// assert_eq!(plaintext_vector.plaintext_count(), PlaintextCount(3));
    /// engine.destroy(plaintext_vector)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_plaintext_vector(
        &mut self,
        input: &[u64],
    ) -> Result<PlaintextVector64, PlaintextVectorCreationError<Self::EngineError>> {
        if input.is_empty() {
            return Err(PlaintextVectorCreationError::EmptyInput);
        }
        Ok(unsafe { self.create_plaintext_vector_unchecked(input) })
    }

    unsafe fn create_plaintext_vector_unchecked(&mut self, input: &[u64]) -> PlaintextVector64 {
        PlaintextVector64(ImplPlaintextList::from_container(input.to_vec()))
    }
}
