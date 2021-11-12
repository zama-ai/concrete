use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{CleartextVector32, CleartextVector64};
use crate::backends::core::private::crypto::encoding::CleartextList as ImplCleartextList;
use crate::specification::engines::{CleartextVectorCreationEngine, CleartextVectorCreationError};

impl CleartextVectorCreationEngine<u32, CleartextVector32> for CoreEngine {
    fn create_cleartext_vector(
        &mut self,
        input: &[u32],
    ) -> Result<CleartextVector32, CleartextVectorCreationError<Self::EngineError>> {
        if input.is_empty() {
            return Err(CleartextVectorCreationError::EmptyInput);
        }
        Ok(unsafe { self.create_cleartext_vector_unchecked(input) })
    }

    unsafe fn create_cleartext_vector_unchecked(&mut self, input: &[u32]) -> CleartextVector32 {
        CleartextVector32(ImplCleartextList::from_container(input.to_vec()))
    }
}

impl CleartextVectorCreationEngine<u64, CleartextVector64> for CoreEngine {
    fn create_cleartext_vector(
        &mut self,
        input: &[u64],
    ) -> Result<CleartextVector64, CleartextVectorCreationError<Self::EngineError>> {
        if input.is_empty() {
            return Err(CleartextVectorCreationError::EmptyInput);
        }
        Ok(unsafe { self.create_cleartext_vector_unchecked(input) })
    }

    unsafe fn create_cleartext_vector_unchecked(&mut self, input: &[u64]) -> CleartextVector64 {
        CleartextVector64(ImplCleartextList::from_container(input.to_vec()))
    }
}
