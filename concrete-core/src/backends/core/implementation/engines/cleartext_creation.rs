use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Cleartext32, Cleartext64};
use crate::backends::core::private::crypto::encoding::Cleartext as ImplCleartext;
use crate::specification::engines::{CleartextCreationEngine, CleartextCreationError};

impl CleartextCreationEngine<u32, Cleartext32> for CoreEngine {
    fn create_cleartext(
        &mut self,
        input: &u32,
    ) -> Result<Cleartext32, CleartextCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_cleartext_unchecked(input) })
    }

    unsafe fn create_cleartext_unchecked(&mut self, input: &u32) -> Cleartext32 {
        Cleartext32(ImplCleartext(*input))
    }
}

impl CleartextCreationEngine<u64, Cleartext64> for CoreEngine {
    fn create_cleartext(
        &mut self,
        input: &u64,
    ) -> Result<Cleartext64, CleartextCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_cleartext_unchecked(input) })
    }

    unsafe fn create_cleartext_unchecked(&mut self, input: &u64) -> Cleartext64 {
        Cleartext64(ImplCleartext(*input))
    }
}
