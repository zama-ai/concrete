use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize, Seed};
#[cfg(feature = "multithread")]
use rayon::prelude::*;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::glwe::GlweSeededCiphertext;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};

/// A matrix containing a single level of gadget decomposition.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct GgswSeededLevelMatrix<Cont, Scalar> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    level: DecompositionLevel,
    _scalar: std::marker::PhantomData<Scalar>,
    seed: Seed,
}

// tensor_traits!(GgswSeededLevelMatrix);

impl<Cont, Scalar> GgswSeededLevelMatrix<Cont, Scalar> {
    /// Creates a GGSW seeded level matrix from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        poly_size: PolynomialSize,
        glwe_size: GlweSize,
        level: DecompositionLevel,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => poly_size.0);
        Self {
            tensor,
            poly_size,
            glwe_size,
            level,
            _scalar: Default::default(),
            seed,
        }
    }

    /// Returns the size of the GLWE ciphertexts composing the GGSW level matrix.
    ///
    /// This is also the number of columns of the expanded matrix (assuming it is a matrix of
    ///  polynomials), as well as the number of rows of the matrix.
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
        Scalar: Numeric,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .enumerate()
            .map(move |(i, tens)| {
                let seed = Seed {
                    seed: self.seed.seed,
                    shift: self.seed.shift
                        + i * self.glwe_size.to_glwe_dimension().0
                            * self.polynomial_size().0
                            * Scalar::BITS
                            / 8,
                };
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    self.level,
                    self.glwe_size.to_glwe_dimension(),
                    seed,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed rows of the matrix.
    pub fn row_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GgswSeededLevelRow<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        Scalar: Numeric,
    {
        let chunks_size = self.poly_size.0;
        let glwe_dimension = self.glwe_size.to_glwe_dimension();
        let polynomial_size = self.polynomial_size();
        let level = self.level;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tens)| {
                let row_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift + i * glwe_dimension.0 * polynomial_size.0 * Scalar::BITS / 8,
                };
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    level,
                    glwe_dimension,
                    row_seed,
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
        <Self as AsMutTensor>::Element: Numeric + Send + Sync,
    {
        let chunks_size = self.poly_size.0;
        let glwe_dimension = self.glwe_size.to_glwe_dimension();
        let polynomial_size = self.polynomial_size();
        let level = self.level;
        let seed = self.seed;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tens)| {
                let seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + i * glwe_dimension.0
                            * polynomial_size.0
                            * <Self as AsMutTensor>::Element::BITS
                            / 8,
                };
                GgswSeededLevelRow::from_container(
                    tens.into_container(),
                    level,
                    glwe_dimension,
                    seed,
                )
            })
    }
}

impl<Element, Cont, Scalar> AsRefTensor for GgswSeededLevelMatrix<Cont, Scalar>
where
    Cont: AsRefSlice<Element = Element>,
{
    type Element = Element;
    type Container = Cont;
    fn as_tensor(&self) -> &Tensor<Self::Container> {
        &self.tensor
    }
}

impl<Element, Cont, Scalar> AsMutTensor for GgswSeededLevelMatrix<Cont, Scalar>
where
    Cont: AsMutSlice<Element = Element>,
{
    type Element = Element;
    type Container = Cont;
    fn as_mut_tensor(&mut self) -> &mut Tensor<<Self as AsMutTensor>::Container> {
        &mut self.tensor
    }
}

impl<Cont, Scalar> IntoTensor for GgswSeededLevelMatrix<Cont, Scalar>
where
    Cont: AsRefSlice,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}

/// A row of a GGSW level matrix.
pub struct GgswSeededLevelRow<Cont> {
    tensor: Tensor<Cont>,
    level: DecompositionLevel,
    glwe_dimension: GlweDimension,
    seed: Seed,
}

tensor_traits!(GgswSeededLevelRow);

impl<Cont> GgswSeededLevelRow<Cont> {
    /// Creates an Rgsw seeded level row from an arbitrary container.
    pub fn from_container(
        cont: Cont,
        level: DecompositionLevel,
        glwe_dimension: GlweDimension,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        Self {
            tensor: Tensor::from_container(cont),
            level,
            glwe_dimension,
            seed,
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
        )
    }
}
