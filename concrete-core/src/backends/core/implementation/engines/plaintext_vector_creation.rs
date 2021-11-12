use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{PlaintextVector32, PlaintextVector64};
use crate::backends::core::private::crypto::encoding::PlaintextList as ImplPlaintextList;
use crate::specification::engines::{PlaintextVectorCreationEngine, PlaintextVectorCreationError};

impl PlaintextVectorCreationEngine<u32, PlaintextVector32> for CoreEngine {
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

impl PlaintextVectorCreationEngine<u64, PlaintextVector64> for CoreEngine {
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
