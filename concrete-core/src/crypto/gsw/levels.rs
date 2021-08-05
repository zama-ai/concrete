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
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the index of the level corresponding to this matrix.
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Returns an iterator over the borrowed rows of the matrix.
    pub fn row_iter(&self) -> impl Iterator<Item = GswLevelRow<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.lwe_size.0)
            .map(move |tens| GswLevelRow::from_container(tens.into_container(), self.level))
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
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
    /// Creates an Rgsw level row from an arbitrary container.
    pub fn from_container(cont: Cont, level: DecompositionLevel) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        GswLevelRow { tensor, level }
    }

    /// Returns the size of the lwe ciphertexts composing this level row.
    pub fn lwe_size(&self) -> LweSize
    where
        Self: AsRefTensor,
    {
        LweSize(self.as_tensor().len())
    }

    /// Returns the index of the level corresponding to this row.
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Consumes the row and returns its container wrapped into an `LweCiphertext`.
    pub fn into_lwe(self) -> LweCiphertext<Cont> {
        LweCiphertext::from_container(self.tensor.into_container())
    }
}
