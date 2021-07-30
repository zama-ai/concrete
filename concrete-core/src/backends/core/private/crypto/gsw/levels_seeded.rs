use concrete_commons::parameters::{LweDimension, LweSize};
use concrete_csprng::RandomGenerator;
#[cfg(feature = "multithread")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backends::core::private::{
    crypto::lwe::LweSeededCiphertext,
    math::{
        decomposition::DecompositionLevel,
        tensor::{ck_dim_eq, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor},
    },
};

/// A matrix containing a single level of gadget decomposition.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct GswSeededLevelMatrix<Cont> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    level: DecompositionLevel,
    seed: u128,
    shift: usize,
}

tensor_traits!(GswSeededLevelMatrix);

impl<Cont> GswSeededLevelMatrix<Cont> {
    /// Creates a GSW seeded level matrix from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        level: DecompositionLevel,
        seed: u128,
        shift: usize,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_eq!(tensor.as_slice().len() => lwe_size.0);
        GswSeededLevelMatrix {
            tensor,
            lwe_size,
            level,
            seed,
            shift,
        }
    }

    /// Returns the size of the LWE ciphertexts composing the seeded GSW level matrix.
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
    pub fn row_iter(
        &self,
    ) -> impl Iterator<Item = GswSeededLevelRow<&<Self as AsRefTensor>::Element>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor().iter().enumerate().map(move |(i, value)| {
            GswSeededLevelRow::from_scalar(
                value,
                self.level,
                self.lwe_size().to_lwe_dimension(),
                self.seed,
                i,
            )
        })
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
    pub fn row_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GswSeededLevelRow<&mut <Self as AsRefTensor>::Element>>
    where
        Self: AsMutTensor,
    {
        let level = self.level;
        let seed = self.seed;
        let lwe_dimension = self.lwe_size().to_lwe_dimension();
        self.as_mut_tensor()
            .iter_mut()
            .enumerate()
            .map(move |(i, value)| {
                GswSeededLevelRow::from_scalar(value, level, lwe_dimension, seed, i)
            })
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
    ) -> impl IndexedParallelIterator<Item = GswSeededLevelRow<&mut <Self as AsRefTensor>::Element>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Send + Sync,
    {
        let level = self.level;
        let seed = self.seed;
        let lwe_dimension = self.lwe_size().to_lwe_dimension();
        self.as_mut_tensor()
            .par_iter_mut()
            .enumerate()
            .map(move |(i, value)| {
                GswSeededLevelRow::from_scalar(value, level, lwe_dimension, seed, i)
            })
    }
}

/// A row of a seeded GSW level matrix.
pub struct GswSeededLevelRow<Scalar> {
    value: Scalar,
    level: DecompositionLevel,
    lwe_dimension: LweDimension,
    seed: u128,
    shift: usize,
}

impl<Scalar> GswSeededLevelRow<Scalar> {
    /// Creates a GSW level row from an arbitrary container.
    pub fn allocate(value: Scalar, level: DecompositionLevel, lwe_dimension: LweDimension) -> Self {
        GswSeededLevelRow {
            value,
            level,
            lwe_dimension,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }

    pub fn from_scalar(
        value: Scalar,
        level: DecompositionLevel,
        lwe_dimension: LweDimension,
        seed: u128,
        shift: usize,
    ) -> Self {
        GswSeededLevelRow {
            value,
            level,
            lwe_dimension,
            seed,
            shift,
        }
    }

    /// Returns the size of the glwe ciphertext composing this level row.
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_dimension.to_lwe_size()
    }

    /// Returns the index of the level corresponding to this row.
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Consumes the row and returns its value wrapped into an `LweSeededCiphertext`.
    pub fn into_seeded_lwe(self) -> LweSeededCiphertext<Scalar> {
        LweSeededCiphertext::from_scalar(self.value, self.lwe_dimension, self.seed, self.shift)
    }
}
