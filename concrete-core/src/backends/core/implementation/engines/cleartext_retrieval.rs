use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Cleartext32, Cleartext64};
use crate::specification::engines::{CleartextRetrievalEngine, CleartextRetrievalError};

/// # Description:
/// Implementation of [`CleartextRetrievalEngine`] for [`CoreEngine`] that operates on 32 bits
/// integers.
impl CleartextRetrievalEngine<Cleartext32, u32> for CoreEngine {
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
    /// let output: u32 = engine.retrieve_cleartext(&cleartext)?;
    ///
    /// assert_eq!(output, 3_u32);
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_cleartext(
        &mut self,
        cleartext: &Cleartext32,
    ) -> Result<u32, CleartextRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_cleartext_unchecked(cleartext) })
    }

    unsafe fn retrieve_cleartext_unchecked(&mut self, cleartext: &Cleartext32) -> u32 {
        cleartext.0 .0
    }
}

/// # Description:
/// Implementation of [`CleartextRetrievalEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl CleartextRetrievalEngine<Cleartext64, u64> for CoreEngine {
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
    /// let output: u64 = engine.retrieve_cleartext(&cleartext)?;
    ///
    /// assert_eq!(output, 3_u64);
    /// engine.destroy(cleartext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_cleartext(
        &mut self,
        cleartext: &Cleartext64,
    ) -> Result<u64, CleartextRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_cleartext_unchecked(cleartext) })
    }

    unsafe fn retrieve_cleartext_unchecked(&mut self, cleartext: &Cleartext64) -> u64 {
        cleartext.0 .0
    }
}
