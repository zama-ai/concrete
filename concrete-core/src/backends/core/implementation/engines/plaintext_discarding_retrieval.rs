use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{Plaintext32, Plaintext64};
use crate::specification::engines::{
    PlaintextDiscardingRetrievalEngine, PlaintextDiscardingRetrievalError,
};

/// # Description:
/// Implementation of [`PlaintextDiscardingRetrievalEngine`] for [`CoreEngine`] that operates on 32
/// bits integers.
impl PlaintextDiscardingRetrievalEngine<Plaintext32, u32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u32 << 20;
    /// let mut output = 0_u32;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext32 = engine.create_plaintext(&input)?;
    /// engine.discard_retrieve_plaintext(&mut output, &plaintext)?;
    ///
    /// assert_eq!(output, 3_u32 << 20);
    /// engine.destroy(plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_retrieve_plaintext(
        &mut self,
        output: &mut u32,
        input: &Plaintext32,
    ) -> Result<(), PlaintextDiscardingRetrievalError<Self::EngineError>> {
        unsafe { self.discard_retrieve_plaintext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn discard_retrieve_plaintext_unchecked(
        &mut self,
        output: &mut u32,
        input: &Plaintext32,
    ) {
        *output = input.0 .0;
    }
}

impl PlaintextDiscardingRetrievalEngine<Plaintext64, u64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input = 3_u64 << 20;
    /// let mut output = 0_u64;
    ///
    /// let mut engine = CoreEngine::new()?;
    /// let plaintext: Plaintext64 = engine.create_plaintext(&input)?;
    /// engine.discard_retrieve_plaintext(&mut output, &plaintext)?;
    ///
    /// assert_eq!(output, 3_u64 << 20);
    /// engine.destroy(plaintext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn discard_retrieve_plaintext(
        &mut self,
        output: &mut u64,
        input: &Plaintext64,
    ) -> Result<(), PlaintextDiscardingRetrievalError<Self::EngineError>> {
        unsafe { self.discard_retrieve_plaintext_unchecked(output, input) };
        Ok(())
    }

    unsafe fn discard_retrieve_plaintext_unchecked(
        &mut self,
        output: &mut u64,
        input: &Plaintext64,
    ) {
        *output = input.0 .0;
    }
}
