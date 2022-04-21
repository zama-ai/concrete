use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweCiphertextMutView32, GlweCiphertextMutView64,
    GlweCiphertextView32, GlweCiphertextView64,
};
use crate::backends::core::private::math::tensor::IntoTensor;
use crate::specification::engines::{
    GlweCiphertextConsumingRetrievalEngine, GlweCiphertextConsumingRetrievalError,
};

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying vec of a [`GlweCiphertext32`] consuming it in the process
impl GlweCiphertextConsumingRetrievalEngine<GlweCiphertext32, Vec<u32>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u32; glwe_size.0 * polynomial_size.0];
    /// let original_vec_ptr = owned_container.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: GlweCiphertext32 =
    ///     engine.create_glwe_ciphertext(owned_container, polynomial_size)?;
    /// let retrieved_container = engine.consume_retrieve_glwe_ciphertext(ciphertext)?;
    /// assert_eq!(original_vec_ptr, retrieved_container.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertext32,
    ) -> Result<Vec<u32>, GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertext32,
    ) -> Vec<u32> {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying vec of a [`GlweCiphertext64`] consuming it in the process
impl GlweCiphertextConsumingRetrievalEngine<GlweCiphertext64, Vec<u64>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u64; glwe_size.0 * polynomial_size.0];
    /// let original_vec_ptr = owned_container.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: GlweCiphertext64 =
    ///     engine.create_glwe_ciphertext(owned_container, polynomial_size)?;
    /// let retrieved_container = engine.consume_retrieve_glwe_ciphertext(ciphertext)?;
    /// assert_eq!(original_vec_ptr, retrieved_container.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertext64,
    ) -> Result<Vec<u64>, GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertext64,
    ) -> Vec<u64> {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`GlweCiphertextView32`] consuming it in the process
impl<'data> GlweCiphertextConsumingRetrievalEngine<GlweCiphertextView32<'data>, &'data [u32]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u32; glwe_size.0 * polynomial_size.0];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextView32 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// let retrieved_slice = engine.consume_retrieve_glwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(slice, retrieved_slice);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertextView32<'data>,
    ) -> Result<&'data [u32], GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertextView32<'data>,
    ) -> &'data [u32] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`GlweCiphertextMutView32`] consuming it in the process
impl<'data> GlweCiphertextConsumingRetrievalEngine<GlweCiphertextMutView32<'data>, &'data mut [u32]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u32; glwe_size.0 * polynomial_size.0];
    ///
    /// let slice = &mut owned_container[..];
    /// // Required as we can't borrow a mut slice more than once
    /// let underlying_ptr = slice.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextMutView32 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// let retrieved_slice = engine.consume_retrieve_glwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(underlying_ptr, retrieved_slice.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertextMutView32<'data>,
    ) -> Result<&'data mut [u32], GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertextMutView32<'data>,
    ) -> &'data mut [u32] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`GlweCiphertextView64`] consuming it in the process
impl<'data> GlweCiphertextConsumingRetrievalEngine<GlweCiphertextView64<'data>, &'data [u64]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u64; glwe_size.0 * polynomial_size.0];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextView64 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// let retrieved_slice = engine.consume_retrieve_glwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(slice, retrieved_slice);
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertextView64<'data>,
    ) -> Result<&'data [u64], GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertextView64<'data>,
    ) -> &'data [u64] {
        ciphertext.0.into_tensor().into_container()
    }
}

/// # Description:
/// Implementation of [`LweCiphertextConsumingRetrievalEngine`] for [`CoreEngine`] that returns the
/// underlying slice of a [`GlweCiphertextMutView64`] consuming it in the process
impl<'data> GlweCiphertextConsumingRetrievalEngine<GlweCiphertextMutView64<'data>, &'data mut [u64]>
    for CoreEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = GlweSize(600);
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u64; glwe_size.0 * polynomial_size.0];
    ///
    /// let slice = &mut owned_container[..];
    /// // Required as we can't borrow a mut slice more than once
    /// let underlying_ptr = slice.as_ptr();
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextMutView64 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// let retrieved_slice = engine.consume_retrieve_glwe_ciphertext(ciphertext_view)?;
    /// assert_eq!(underlying_ptr, retrieved_slice.as_ptr());
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn consume_retrieve_glwe_ciphertext(
        &mut self,
        ciphertext: GlweCiphertextMutView64<'data>,
    ) -> Result<&'data mut [u64], GlweCiphertextConsumingRetrievalError<Self::EngineError>> {
        Ok(unsafe { self.consume_retrieve_glwe_ciphertext_unchecked(ciphertext) })
    }

    unsafe fn consume_retrieve_glwe_ciphertext_unchecked(
        &mut self,
        ciphertext: GlweCiphertextMutView64<'data>,
    ) -> &'data mut [u64] {
        ciphertext.0.into_tensor().into_container()
    }
}
