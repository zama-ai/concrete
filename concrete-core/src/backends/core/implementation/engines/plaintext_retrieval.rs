use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Plaintext32, Plaintext64};
use crate::specification::engines::{PlaintextRetrievalEngine, PlaintextRetrievalError};

/// # Description:
/// Implementation of [`PlaintextRetrievalEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl PlaintextRetrievalEngine<Plaintext32, u32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u32 << 20;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext32 = engine.create_plaintext(&input)?;
    /// let output: u32 = engine.retrieve_plaintext(&plaintext)?;
    ///
    /// assert_eq!(output, 3_u32 << 20);
    /// engine.destroy(plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_plaintext(
        &mut self,
        plaintext: &Plaintext32,
    ) -> Result<u32, PlaintextRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_plaintext_unchecked(plaintext) })
    }

    unsafe fn retrieve_plaintext_unchecked(&mut self, plaintext: &Plaintext32) -> u32 {
        plaintext.0 .0
    }
}

/// # Description:
/// Implementation of [`PlaintextRetrievalEngine`] for [`CoreEngine`] that operates on 64 bits
/// integers.
impl PlaintextRetrievalEngine<Plaintext64, u64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u64 << 20;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext64 = engine.create_plaintext(&input)?;
    /// let output: u64 = engine.retrieve_plaintext(&plaintext)?;
    ///
    /// assert_eq!(output, 3_u64 << 20);
    /// engine.destroy(plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn retrieve_plaintext(
        &mut self,
        plaintext: &Plaintext64,
    ) -> Result<u64, PlaintextRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.retrieve_plaintext_unchecked(plaintext) })
    }

    unsafe fn retrieve_plaintext_unchecked(&mut self, plaintext: &Plaintext64) -> u64 {
        plaintext.0 .0
    }
}
