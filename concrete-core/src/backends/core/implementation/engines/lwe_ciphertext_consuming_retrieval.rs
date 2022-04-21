use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweCiphertextMutView32, LweCiphertextMutView64,
    LweCiphertextView32, LweCiphertextView64,
};
use crate::backends::core::private::math::tensor::IntoTensor;
use crate::specification::engines::{
    LweCiphertextConsumingRetrievalEngine, LweCiphertextConsumingRetrievalError,
};

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying vec of a [`LweCiphertext32`] consuming it in the process
impl LweCiphertextConsumingRetrievalEngine<LweCiphertext32, Vec<u32>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u32; lwe_size.0];
    /// let original_vec_ptr = owned_container.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: LweCiphertext32 = engine.create_lwe_ciphertext(owned_container)?;
    /// let retrieved_container = engine.consume_retrieve_lwe_ciphertext(ciphertext)?;
    /// assert_eq!(original_vec_ptr, retrieved_container.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertext32,
    ) -> Result<Vec<u32>, LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertext32,
    ) -> Vec<u32> {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying vec of a [`LweCiphertext64`] consuming it in the process
impl LweCiphertextConsumingRetrievalEngine<LweCiphertext64, Vec<u64>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u64; lwe_size.0];
    /// let original_vec_ptr = owned_container.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: LweCiphertext64 = engine.create_lwe_ciphertext(owned_container)?;
    /// let retrieved_container = engine.consume_retrieve_lwe_ciphertext(ciphertext)?;
    /// assert_eq!(original_vec_ptr, retrieved_container.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertext64,
    ) -> Result<Vec<u64>, LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertext64,
    ) -> Vec<u64> {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`LweCiphertextView32`] consuming it in the process
impl<'data> LweCiphertextConsumingRetrievalEngine<LweCiphertextView32<'data>, &'data [u32]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u32; lwe_size.0];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextView32 = engine.create_lwe_ciphertext(slice)?;
    /// let retrieved_slice = engine.consume_retrieve_lwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(slice, retrieved_slice);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertextView32<'data>,
    ) -> Result<&'data [u32], LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertextView32<'data>,
    ) -> &'data [u32] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`LweCiphertextMutView32`] consuming it in the process
impl<'data> LweCiphertextConsumingRetrievalEngine<LweCiphertextMutView32<'data>, &'data mut [u32]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u32; lwe_size.0];
    ///
    /// let slice = &mut owned_container[..];
    /// // Required as we can't borrow a mut slice more than once
    /// let underlying_ptr = slice.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextMutView32 = engine.create_lwe_ciphertext(slice)?;
    /// let retrieved_slice = engine.consume_retrieve_lwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(underlying_ptr, retrieved_slice.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertextMutView32<'data>,
    ) -> Result<&'data mut [u32], LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertextMutView32<'data>,
    ) -> &'data mut [u32] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`LweCiphertextView64`] consuming it in the process
impl<'data> LweCiphertextConsumingRetrievalEngine<LweCiphertextView64<'data>, &'data [u64]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u64; lwe_size.0];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextView64 = engine.create_lwe_ciphertext(slice)?;
    /// let retrieved_slice = engine.consume_retrieve_lwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(slice, retrieved_slice);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertextView64<'data>,
    ) -> Result<&'data [u64], LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertextView64<'data>,
    ) -> &'data [u64] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`LweCiphertextMutView64`] consuming it in the process
impl<'data> LweCiphertextConsumingRetrievalEngine<LweCiphertextMutView64<'data>, &'data mut [u64]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let lwe_size = LweSize(128);
    /// let mut owned_container = vec![0_u64; lwe_size.0];
    ///
    /// let slice = &mut owned_container[..];
    /// // Required as we can't borrow a mut slice more than once
    /// let underlying_ptr = slice.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextMutView64 = engine.create_lwe_ciphertext(slice)?;
    /// let retrieved_slice = engine.consume_retrieve_lwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(underlying_ptr, retrieved_slice.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_lwe_ciphertext(
        &mut self,
        ciphertext: LweCiphertextMutView64<'data>,
    ) -> Result<&'data mut [u64], LweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_lwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: LweCiphertextMutView64<'data>,
    ) -> &'data mut [u64] {
        ciphertext.0.into_tensor().into_container()
    }
}
