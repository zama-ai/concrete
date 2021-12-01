use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Cleartext32, Cleartext64};
use crate::backends::core::private::crypto::encoding::Cleartext as ImplCleartext;
use crate::specification::engines::{CleartextCreationEngine, CleartextCreationError};

/// # Description:
/// Implementation of [`CleartextCreationEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl CleartextCreationEngine<u32, Cleartext32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let input: u32 = 3;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext: Cleartext32 = engine.create_cleartext(&input)?;
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
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

/// # Description:
/// Implementation of [`CleartextCreationEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl CleartextCreationEngine<u64, Cleartext64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let input: u64 = 3;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let cleartext: Cleartext64 = engine.create_cleartext(&input)?;
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
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
