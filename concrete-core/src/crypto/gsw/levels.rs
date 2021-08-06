use crate::crypto::lwe::LweCiphertext;
use crate::math::decomposition::DecompositionLevel;
use crate::math::tensor::{AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use crate::{ck_dim_eq, tensor_traits};
use concrete_commons::parameters::LweSize;
#[cfg(feature = "multithread")]
use rayon::prelude::*;

/// A matrix containing a single level of gadget decomposition.
pub struct GswLevelMatrix<Cont> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    level: DecompositionLevel,
}

tensor_traits!(GswLevelMatrix);

impl<Cont> GswLevelMatrix<Cont> {
    /// Creates a GSW level matrix from an arbitrary container.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// assert_eq!(level_matrix.lwe_size(), LweSize(7));
    /// assert_eq!(level_matrix.decomposition_level(), DecompositionLevel(1));
    /// ```
    pub fn from_container(cont: Cont, lwe_size: LweSize, level: DecompositionLevel) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_eq!(tensor.as_slice().len() => lwe_size.0 * lwe_size.0);
        GswLevelMatrix {
            tensor,
            lwe_size,
            level,
        }
    }

    /// Returns the size of the LWE ciphertexts composing the GGSW level matrix.
    ///
    /// This is also the number of columns of the matrix
    /// , as well as its number of rows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// assert_eq!(level_matrix.lwe_size(), LweSize(7));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the index of the level corresponding to this matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// assert_eq!(level_matrix.decomposition_level(), DecompositionLevel(1));
    /// ```
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Returns an iterator over the borrowed rows of the matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// for row in level_matrix.row_iter() {
    ///     assert_eq!(row.lwe_size(), LweSize(7));
    /// }
    /// assert_eq!(level_matrix.row_iter().count(), 7);
    /// ```
    pub fn row_iter(&self) -> impl Iterator<Item = GswLevelRow<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.lwe_size.0)
            .map(move |tens| GswLevelRow::from_container(tens.into_container(), self.level))
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// for mut row in level_matrix.row_iter_mut() {
    ///     row.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert!(level_matrix.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(level_matrix.row_iter_mut().count(), 7);
    /// ```
    pub fn row_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GswLevelRow<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.lwe_size.0;
        let level = self.level;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |tens| GswLevelRow::from_container(tens.into_container(), level))
    }

    /// Returns a parallel iterator over the mutably borrowed rows of the matrix.
    ///
    /// # Note
    ///
    /// This method uses _rayon_ internally, and is hidden behind the "multithread" feature
    /// gate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelMatrix;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    /// let mut level_matrix =
    ///     GswLevelMatrix::from_container(vec![0 as u8; 7 * 7], LweSize(7), DecompositionLevel(1));
    /// level_matrix.par_row_iter_mut().for_each(|mut row| {
    ///     row.as_mut_tensor().fill_with_element(9);
    /// });
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_row_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GswLevelRow<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Send + Sync,
    {
        let chunks_size = self.lwe_size.0;
        let level = self.level;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .map(move |tens| GswLevelRow::from_container(tens.into_container(), level))
    }
}

/// A row of a GSW level matrix.
pub struct GswLevelRow<Cont> {
    tensor: Tensor<Cont>,
    level: DecompositionLevel,
}

tensor_traits!(GswLevelRow);

impl<Cont> GswLevelRow<Cont> {
    /// Creates a GSW level row from an arbitrary container.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelRow;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_row = GswLevelRow::from_container(vec![0 as u8; 7], DecompositionLevel(1));
    /// assert_eq!(level_row.lwe_size(), LweSize(7));
    /// assert_eq!(level_row.decomposition_level(), DecompositionLevel(1));
    /// ```
    pub fn from_container(cont: Cont, level: DecompositionLevel) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        GswLevelRow { tensor, level }
    }

    /// Returns the size of the lwe ciphertexts composing this level row.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelRow;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_row = GswLevelRow::from_container(vec![0 as u8; 7], DecompositionLevel(1));
    /// assert_eq!(level_row.lwe_size(), LweSize(7));
    /// ```
    pub fn lwe_size(&self) -> LweSize
    where
        Self: AsRefTensor,
    {
        LweSize(self.as_tensor().len())
    }

    /// Returns the index of the level corresponding to this row.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelRow;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_row = GswLevelRow::from_container(vec![0 as u8; 7], DecompositionLevel(1));
    /// assert_eq!(level_row.decomposition_level(), DecompositionLevel(1));
    /// ```
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Consumes the row and returns its container wrapped into an `LweCiphertext`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::crypto::gsw::GswLevelRow;
    /// use concrete_core::math::decomposition::DecompositionLevel;
    /// let level_row = GswLevelRow::from_container(vec![0 as u8; 7], DecompositionLevel(1));
    /// let lwe = level_row.into_lwe();
    /// assert_eq!(lwe.lwe_size(), LweSize(7));
    /// ```
    pub fn into_lwe(self) -> LweCiphertext<Cont> {
        LweCiphertext::from_container(self.tensor.into_container())
    }
}
