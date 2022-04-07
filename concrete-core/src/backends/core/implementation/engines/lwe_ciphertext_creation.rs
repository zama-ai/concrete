use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    LweCiphertext32, LweCiphertext64, LweCiphertextMutView32, LweCiphertextMutView64,
    LweCiphertextView32, LweCiphertextView64,
};
use crate::backends::core::private::crypto::lwe::LweCiphertext as ImplLweCiphertext;
use crate::specification::engines::{LweCiphertextCreationEngine, LweCiphertextCreationError};

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns an
/// [`LweCiphertext32`].
impl LweCiphertextCreationEngine<Vec<u32>, LweCiphertext32> for CoreEngine {
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
    /// let owned_container = vec![0_u32; lwe_size.0];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: LweCiphertext32 = engine.create_lwe_ciphertext(owned_container)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: Vec<u32>,
    ) -> Result<LweCiphertext32, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(&mut self, container: Vec<u32>) -> LweCiphertext32 {
        LweCiphertext32(ImplLweCiphertext::from_container(container))
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns an
/// [`LweCiphertext64`].
impl LweCiphertextCreationEngine<Vec<u64>, LweCiphertext64> for CoreEngine {
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
    /// let owned_container = vec![0_u64; lwe_size.0];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: LweCiphertext64 = engine.create_lwe_ciphertext(owned_container)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: Vec<u64>,
    ) -> Result<LweCiphertext64, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(&mut self, container: Vec<u64>) -> LweCiphertext64 {
        LweCiphertext64(ImplLweCiphertext::from_container(container))
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns an immutable
/// [`LweCiphertextView32`] that does not own its memory.
impl<'data> LweCiphertextCreationEngine<&'data [u32], LweCiphertextView32<'data>> for CoreEngine {
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
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: &'data [u32],
    ) -> Result<LweCiphertextView32<'data>, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(
        &mut self,
        container: &'data [u32],
    ) -> LweCiphertextView32<'data> {
        LweCiphertextView32(ImplLweCiphertext::from_container(container))
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns a mutable
/// [`LweCiphertextMutView32`] that does not own its memory.
impl<'data> LweCiphertextCreationEngine<&'data mut [u32], LweCiphertextMutView32<'data>>
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
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextMutView32 = engine.create_lwe_ciphertext(slice)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: &'data mut [u32],
    ) -> Result<LweCiphertextMutView32<'data>, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(
        &mut self,
        container: &'data mut [u32],
    ) -> LweCiphertextMutView32<'data> {
        LweCiphertextMutView32(ImplLweCiphertext::from_container(container))
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns an immutable
/// [`LweCiphertextView64`] that does not own its memory.
impl<'data> LweCiphertextCreationEngine<&'data [u64], LweCiphertextView64<'data>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let mut owned_container = vec![0_u64; 128];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextView64 = engine.create_lwe_ciphertext(slice)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: &'data [u64],
    ) -> Result<LweCiphertextView64<'data>, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(
        &mut self,
        container: &'data [u64],
    ) -> LweCiphertextView64<'data> {
        LweCiphertextView64(ImplLweCiphertext::from_container(container))
    }
}

/// # Description:
/// Implementation of [`LweCiphertextCreationEngine`] for [`CoreEngine`] which returns a mutable
/// [`LweCiphertextMutView64`] that does not own its memory.
impl<'data> LweCiphertextCreationEngine<&'data mut [u64], LweCiphertextMutView64<'data>>
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
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: LweCiphertextMutView64 = engine.create_lwe_ciphertext(slice)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_lwe_ciphertext(
        &mut self,
        container: &'data mut [u64],
    ) -> Result<LweCiphertextMutView64<'data>, LweCiphertextCreationError<Self::EngineError>> {
        LweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(container.len())?;
        Ok(unsafe { self.create_lwe_ciphertext_unchecked(container) })
    }

    unsafe fn create_lwe_ciphertext_unchecked(
        &mut self,
        container: &'data mut [u64],
    ) -> LweCiphertextMutView64<'data> {
        LweCiphertextMutView64(ImplLweCiphertext::from_container(container))
    }
}
