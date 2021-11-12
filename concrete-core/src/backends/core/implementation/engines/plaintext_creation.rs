use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Plaintext32, Plaintext64};
use crate::backends::core::private::crypto::encoding::Plaintext as ImplPlaintext;
use crate::specification::engines::{PlaintextCreationEngine, PlaintextCreationError};

impl PlaintextCreationEngine<u32, Plaintext32> for CoreEngine {
    fn create_plaintext(
        &mut self,
        input: &u32,
    ) -> Result<Plaintext32, PlaintextCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_plaintext_unchecked(input) })
    }

    unsafe fn create_plaintext_unchecked(&mut self, input: &u32) -> Plaintext32 {
        Plaintext32(ImplPlaintext(*input))
    }
}

impl PlaintextCreationEngine<u64, Plaintext64> for CoreEngine {
    fn create_plaintext(
        &mut self,
        input: &u64,
    ) -> Result<Plaintext64, PlaintextCreationError<Self::EngineError>> {
        Ok(unsafe { self.create_plaintext_unchecked(input) })
    }

    unsafe fn create_plaintext_unchecked(&mut self, input: &u64) -> Plaintext64 {
        Plaintext64(ImplPlaintext(*input))
    }
}
