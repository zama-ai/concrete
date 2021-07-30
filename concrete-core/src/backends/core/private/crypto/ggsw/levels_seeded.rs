use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
#[cfg(feature = "multithread")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::backends::core::private::{
    crypto::glwe::GlweSeededCiphertext,
    math::{
        decomposition::DecompositionLevel,
        tensor::{ck_dim_div, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor},
    },
};

/// A matrix containing a single level of gadget decomposition.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
pub struct GgswSeededLevelMatrix<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    level: DecompositionLevel,
    seed: u128,
}

tensor_traits!(GgswSeededLevelMatrix);

impl<Cont> GgswSeededLevelMatrix<Cont> {
    /// Creates a GGSW seeded level matrix from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        poly_size: PolynomialSize,
        glwe_size: GlweSize,
        level: DecompositionLevel,
        seed: u128,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => poly_size.0);
        Self {
            tensor,
            poly_size,
            glwe_size: glwe_size,
            level,
            seed,
        }
    }

    /// Returns the size of the GLWE ciphertexts composing the GGSW level matrix.
    ///
    /// This is also the number of columns of the expanded matrix (assuming it is a matrix of polynomials)
    /// , as well as the number of rows of the matrix.
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the index of the level corresponding to this matrix.
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Returns the size of the polynomials of the current ciphertext.
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns an iterator over the borrowed rows of the matrix.
    pub fn row_iter(
        &self,
    ) -> impl Iterator<Item = GgswSeededLevelRow<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .enumerate()
            .map(move |(i, tens)| {
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    self.level,
                    self.glwe_size.to_glwe_dimension(),
                    self.seed,
                    i,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
    pub fn row_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GgswSeededLevelRow<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0;
        let glwe_dimension = self.glwe_size.to_glwe_dimension();
        let level = self.level;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tens)| {
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    level,
                    glwe_dimension,
                    seed,
                    i,
                )
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
    ) -> impl IndexedParallelIterator<Item = GgswSeededLevelRow<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Send + Sync,
    {
        let chunks_size = self.poly_size.0;
        let glwe_dimension = self.glwe_size.to_glwe_dimension();
        let level = self.level;
        let seed = self.seed;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tens)| {
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    level,
                    glwe_dimension,
                    seed,
                    i,
                )
            })
    }
}

/// A row of a GGSW level matrix.
pub struct GgswSeededLevelRow<Cont> {
    tensor: Tensor<Cont>,
    level: DecompositionLevel,
    glwe_dimension: GlweDimension,
    seed: u128,
    shift: usize,
}

tensor_traits!(GgswSeededLevelRow);

impl<Cont> GgswSeededLevelRow<Cont> {
    /// Creates an Rgsw seeded level row from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        level: DecompositionLevel,
        glwe_dimension: GlweDimension,
        seed: u128,
        shift: usize,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        Self {
            tensor: Tensor::from_container(cont),
            level,
            glwe_dimension,
            seed,
            shift,
        }
    }

    /// Returns the size of the glwe ciphertext composing this level row.
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_dimension.to_glwe_size()
    }

    /// Returns the index of the level corresponding to this row.
    pub fn decomposition_level(&self) -> DecompositionLevel {
        self.level
    }

    /// Returns the size of the polynomials used in the row.
    pub fn polynomial_size(&self) -> PolynomialSize
    where
        Cont: AsRefSlice,
    {
        PolynomialSize(self.tensor.len())
    }

    /// Consumes the row and returns its container wrapped into an `GlweCiphertext`.
    pub fn into_seeded_glwe(self) -> GlweSeededCiphertext<Cont> {
        GlweSeededCiphertext::from_container(
            self.tensor.into_container(),
            self.glwe_dimension,
            self.seed,
            self.shift,
        )
    }
}
