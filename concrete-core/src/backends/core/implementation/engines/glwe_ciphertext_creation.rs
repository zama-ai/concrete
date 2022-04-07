use crate::backends::core::implementation::engines::CoreEngine;
use crate::backends::core::implementation::entities::{
    GlweCiphertext32, GlweCiphertext64, GlweCiphertextMutView32, GlweCiphertextMutView64,
    GlweCiphertextView32, GlweCiphertextView64,
};
use crate::backends::core::private::crypto::glwe::GlweCiphertext as ImplGlweCiphertext;
use crate::specification::engines::{GlweCiphertextCreationEngine, GlweCiphertextCreationError};
use concrete_commons::parameters::PolynomialSize;

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns a
/// [`GlweCiphertext32`].
impl GlweCiphertextCreationEngine<Vec<u32>, GlweCiphertext32> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::GlweSize;
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
    /// let owned_container = vec![0_u32; glwe_size.0 * polynomial_size.0];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: GlweCiphertext32 =
    ///     engine.create_glwe_ciphertext(owned_container, polynomial_size)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: Vec<u32>,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertext32, GlweCiphertextCreationError<Self::EngineError>> {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: Vec<u32>,
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertext32 {
        GlweCiphertext32(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns a
/// [`GlweCiphertext64`].
impl GlweCiphertextCreationEngine<Vec<u64>, GlweCiphertext64> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::GlweSize;
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
    /// let owned_container = vec![0_u64; glwe_size.0 * polynomial_size.0];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext: GlweCiphertext64 =
    ///     engine.create_glwe_ciphertext(owned_container, polynomial_size)?;
    /// engine.destroy(ciphertext)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: Vec<u64>,
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertext64, GlweCiphertextCreationError<Self::EngineError>> {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: Vec<u64>,
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertext64 {
        GlweCiphertext64(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns an immutable
/// [`GlweCiphertextView32`] that does not own its memory.
impl<'data> GlweCiphertextCreationEngine<&'data [u32], GlweCiphertextView32<'data>> for CoreEngine {
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
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: &'data [u32],
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertextView32<'data>, GlweCiphertextCreationError<Self::EngineError>> {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: &'data [u32],
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertextView32<'data> {
        GlweCiphertextView32(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns a mutable
/// [`GlweCiphertextMutView32`] that does not own its memory.
impl<'data> GlweCiphertextCreationEngine<&'data mut [u32], GlweCiphertextMutView32<'data>>
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
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextMutView32 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: &'data mut [u32],
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertextMutView32<'data>, GlweCiphertextCreationError<Self::EngineError>>
    {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: &'data mut [u32],
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertextMutView32<'data> {
        GlweCiphertextMutView32(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns an immutable
/// [`GlweCiphertextView64`] that does not own its memory.
impl<'data> GlweCiphertextCreationEngine<&'data [u64], GlweCiphertextView64<'data>> for CoreEngine {
    /// # Example:
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // Here we create a container outside of the engine
    /// // Note that the size here is just for demonstration purposes and should not be chosen
    /// // without proper security analysis for production
    /// let glwe_size = 600_usize;
    /// let polynomial_size = PolynomialSize(1024);
    ///
    /// // You have to make sure you size the container properly
    /// let mut owned_container = vec![0_u64; (glwe_size + 1) * polynomial_size.0];
    ///
    /// let slice = &owned_container[..];
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextView64 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: &'data [u64],
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertextView64<'data>, GlweCiphertextCreationError<Self::EngineError>> {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: &'data [u64],
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertextView64<'data> {
        GlweCiphertextView64(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextCreationEngine`] for [`CoreEngine`] which returns a mutable
/// [`GlweCiphertextMutView64`] that does not own its memory.
impl<'data> GlweCiphertextCreationEngine<&'data mut [u64], GlweCiphertextMutView64<'data>>
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
    ///
    /// let mut engine = CoreEngine::new(())?;
    /// let ciphertext_view: GlweCiphertextMutView64 =
    ///     engine.create_glwe_ciphertext(slice, polynomial_size)?;
    /// engine.destroy(ciphertext_view)?;
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn create_glwe_ciphertext(
        &mut self,
        container: &'data mut [u64],
        polynomial_size: PolynomialSize,
    ) -> Result<GlweCiphertextMutView64<'data>, GlweCiphertextCreationError<Self::EngineError>>
    {
        GlweCiphertextCreationError::<Self::EngineError>::perform_generic_checks(
            container.len(),
            polynomial_size,
        )?;
        Ok(unsafe { self.create_glwe_ciphertext_unchecked(container, polynomial_size) })
    }

    unsafe fn create_glwe_ciphertext_unchecked(
        &mut self,
        container: &'data mut [u64],
        polynomial_size: PolynomialSize,
    ) -> GlweCiphertextMutView64<'data> {
        GlweCiphertextMutView64(ImplGlweCiphertext::from_container(
            container,
            polynomial_size,
        ))
    }
}
