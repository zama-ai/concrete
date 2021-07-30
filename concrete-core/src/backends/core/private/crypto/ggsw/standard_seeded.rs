use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
};
use concrete_csprng::RandomGenerator;
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};

use crate::backends::core::private::{
    crypto::{glwe::GlweSeededList, secret::generators::EncryptionRandomGenerator},
    math::{
        decomposition::DecompositionLevel,
        tensor::{
            ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor,
            IntoTensor, Tensor,
        },
        torus::UnsignedTorus,
    },
};

use super::{GgswSeededLevelMatrix, StandardGgswCiphertext};

/// A GGSW seeded ciphertext.
pub struct StandardGgswSeededCiphertext<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    decomp_base_log: DecompositionBaseLog,
    seed: u128,
    shift: usize,
}

tensor_traits!(StandardGgswSeededCiphertext);

impl<Scalar> StandardGgswSeededCiphertext<Vec<Scalar>> {
    /// Allocates a new GGSW ciphertext whose coefficients are all `value`.
    pub fn allocate(
        value: Scalar,
        poly_size: PolynomialSize,
        glwe_size: GlweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Copy,
    {
        Self {
            tensor: Tensor::from_container(vec![value; decomp_level.0 * glwe_size.0 * poly_size.0]),
            poly_size,
            glwe_size,
            decomp_base_log,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }
}

impl<Cont> StandardGgswSeededCiphertext<Cont> {
    /// Creates a ggsw seeded ciphertext from an existing container.
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
        seed: u128,
        shift: usize,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => poly_size.0, glwe_size.0);
        Self {
            tensor,
            glwe_size,
            poly_size,
            decomp_base_log,
            seed,
            shift,
        }
    }

    /// Returns the size of the glwe ciphertexts composing the ggsw ciphertext.
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    pub(crate) fn get_seed(&self) -> u128 {
        self.seed
    }

    pub(crate) fn get_shift(&self) -> usize {
        self.shift
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.glwe_size.0,
            self.poly_size.0
        );
        DecompositionLevelCount(self.as_tensor().len() / (self.glwe_size.0 * self.poly_size.0))
    }

    /// Returns the size of the polynomials used in the ciphertext.
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed list composed of all the GLWE seeded ciphertexts composing current ciphertext.
    pub fn as_glwe_seeded_list<Scalar>(&self) -> GlweSeededList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        GlweSeededList::from_container(
            self.as_tensor().as_slice(),
            self.glwe_size.to_glwe_dimension(),
            self.poly_size,
            self.seed,
            self.shift,
        )
    }

    /// Returns a mutably borrowed `GlweSeededList` composed of all the GLWE ciphertext composing
    /// current ciphertext.
    pub fn as_mut_glwe_seeded_list<Scalar>(&mut self) -> GlweSeededList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let dimension = self.glwe_size.to_glwe_dimension();
        let size = self.poly_size;
        let seed = self.seed;
        let shift = self.shift;
        GlweSeededList::from_container(
            self.as_mut_tensor().as_mut_slice(),
            dimension,
            size,
            seed,
            shift,
        )
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns an iterator over borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GgswSeededLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    self.seed + (index * self.glwe_size().0) as u128,
                )
            })
    }

    /// Returns an iterator over mutably borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GgswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    seed + (index * glwe_size.0) as u128,
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed level seeded matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GgswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    seed + (index * glwe_size.0) as u128,
                )
            })
    }

    /// Returns the ciphertext as a full fledged GgswCiphertext
    pub fn expand_into<Scalar, OutCont>(&self, output: &mut StandardGgswCiphertext<OutCont>)
    where
        Scalar: UnsignedTorus,
        StandardGgswCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
    {
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        generator.shift(
            Scalar::BITS / 8
                * self.glwe_size().to_glwe_dimension().0
                * self.decomposition_level_count().0
                * self.glwe_size().to_glwe_dimension().0
                * self.polynomial_size().0
                * self.shift,
        );
        output
            .as_mut_glwe_list()
            .ciphertext_iter_mut()
            .zip(self.as_glwe_seeded_list().ciphertext_iter())
            .for_each(|(mut glwe_out, glwe_in)| {
                let (mut output_body, mut output_mask) = glwe_out.get_mut_body_and_mask();

                // generate a uniformly random mask
                generator.fill_tensor_with_random_mask(output_mask.as_mut_tensor());

                output_body
                    .as_mut_tensor()
                    .as_mut_slice()
                    .clone_from_slice(glwe_in.into_tensor().as_slice());
            });
    }
}
