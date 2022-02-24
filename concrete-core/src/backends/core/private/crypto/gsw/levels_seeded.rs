use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{LweDimension, LweSize, Seed};
#[cfg(feature = "multithread")]
use rayon::prelude::*;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::lwe::LweSeededCiphertext;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::tensor::{
    ck_dim_eq, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

/// A matrix containing a single level of gadget decomposition.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct GswSeededLevelMatrix<Cont> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    level: DecompositionLevel,
    seed: Seed,
}

tensor_traits!(GswSeededLevelMatrix);

impl<Cont> GswSeededLevelMatrix<Cont> {
    /// Creates a GSW seeded level matrix from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        level: DecompositionLevel,
        seed: Seed,
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
        }
    }

    /// Returns the size of the LWE ciphertexts composing the seeded GSW level matrix.
    ///
    /// This is also the number of columns of the matrix,
    /// as well as its number of rows.
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
        <Self as AsRefTensor>::Element: Numeric,
    {
        self.as_tensor().iter().enumerate().map(move |(i, value)| {
            let level_seed = Seed {
                seed: self.seed.seed,
                shift: self.seed.shift
                    + i * <Self as AsRefTensor>::Element::BITS / 8
                        * self.lwe_size.to_lwe_dimension().0,
            };
            GswSeededLevelRow::from_scalar(
                value,
                self.level,
                self.lwe_size().to_lwe_dimension(),
                level_seed,
            )
        })
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
    pub fn row_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GswSeededLevelRow<&mut <Self as AsRefTensor>::Element>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        let level = self.level;
        let seed = self.seed.seed;
        let shift = self.seed.shift;
        let lwe_dimension = self.lwe_size().to_lwe_dimension();
        self.as_mut_tensor()
            .iter_mut()
            .enumerate()
            .map(move |(i, value)| {
                let level_seed = Seed {
                    seed,
                    shift: shift + i * <Self as AsMutTensor>::Element::BITS / 8 * lwe_dimension.0,
                };
                GswSeededLevelRow::from_scalar(value, level, lwe_dimension, level_seed)
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
        <Self as AsMutTensor>::Element: Numeric + Send + Sync,
    {
        let level = self.level;
        let seed = self.seed.seed;
        let shift = self.seed.shift;
        let lwe_dimension = self.lwe_size().to_lwe_dimension();
        self.as_mut_tensor()
            .par_iter_mut()
            .enumerate()
            .map(move |(i, value)| {
                let level_seed = Seed {
                    seed,
                    shift: shift + i * <Self as AsMutTensor>::Element::BITS / 8 * lwe_dimension.0,
                };
                GswSeededLevelRow::from_scalar(value, level, lwe_dimension, level_seed)
            })
    }
}

/// A row of a seeded GSW level matrix.
pub struct GswSeededLevelRow<Scalar> {
    value: Scalar,
    level: DecompositionLevel,
    lwe_dimension: LweDimension,
    seed: Option<Seed>,
}

impl<Scalar> GswSeededLevelRow<Scalar> {
    /// Creates a GSW level row from an arbitrary container.
    pub fn allocate(value: Scalar, level: DecompositionLevel, lwe_dimension: LweDimension) -> Self {
        GswSeededLevelRow {
            value,
            level,
            lwe_dimension,
            seed: None,
        }
    }

    pub fn from_scalar(
        value: Scalar,
        level: DecompositionLevel,
        lwe_dimension: LweDimension,
        seed: Seed,
    ) -> Self {
        GswSeededLevelRow {
            value,
            level,
            lwe_dimension,
            seed: Some(seed),
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
        LweSeededCiphertext::from_scalar(self.value, self.lwe_dimension, self.seed.unwrap())
    }

    pub fn as_mut_scalar(&mut self) -> &mut Scalar {
        &mut self.value
    }
}
