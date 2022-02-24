use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::glwe::GlweSeededList;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};

use super::{GgswSeededLevelMatrix, StandardGgswCiphertext};

/// A GGSW seeded ciphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct StandardGgswSeededCiphertext<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    decomp_base_log: DecompositionBaseLog,
    seed: Option<Seed>,
}

tensor_traits!(StandardGgswSeededCiphertext);

impl<Scalar> StandardGgswSeededCiphertext<Vec<Scalar>> {
    /// Allocates a new GGSW ciphertext whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
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
            seed: None,
        }
    }
}

impl<Cont> StandardGgswSeededCiphertext<Cont> {
    /// Creates a ggsw seeded ciphertext from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
        seed: Seed,
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
            seed: Some(seed),
        }
    }

    /// Returns the size of the glwe ciphertexts composing the ggsw ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    pub(crate) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    pub(crate) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// ```
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

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns the size of the polynomials used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed list composed of all the GLWE seeded ciphertexts composing current
    /// ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let list = ggsw.as_glwe_seeded_list();
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_glwe_seeded_list<Scalar>(&self) -> GlweSeededList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        GlweSeededList::from_container(
            self.as_tensor().as_slice(),
            self.glwe_size.to_glwe_dimension(),
            self.poly_size,
            self.seed.unwrap(),
        )
    }

    /// Returns a mutably borrowed `GlweSeededList` composed of all the GLWE ciphertext composing
    /// current ciphertext.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let mut ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut list = ggsw.as_mut_glwe_seeded_list();
    /// list.as_mut_tensor().fill_with_element(0);
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ggsw.as_tensor().iter().for_each(|a| assert_eq!(*a, 0));
    /// ```
    pub fn as_mut_glwe_seeded_list<Scalar>(&mut self) -> GlweSeededList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let dimension = self.glwe_size.to_glwe_dimension();
        let size = self.poly_size;
        let seed = self.seed.unwrap();
        GlweSeededList::from_container(self.as_mut_tensor().as_mut_slice(), dimension, size, seed)
    }

    /// Returns an iterator over borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    ///
    /// let ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 9 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for level_matrix in ggsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     assert_eq!(level_matrix.polynomial_size(), PolynomialSize(9));
    ///     for rlwe in level_matrix.row_iter() {
    ///         assert_eq!(rlwe.glwe_size(), GlweSize(7));
    ///         assert_eq!(rlwe.polynomial_size(), PolynomialSize(9));
    ///     }
    /// }
    /// assert_eq!(ggsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<
        Item = GgswSeededLevelMatrix<
            &[<Self as AsRefTensor>::Element],
            <Self as AsRefTensor>::Element,
        >,
    >
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let seed = Seed {
                    seed: self.seed.unwrap().seed,
                    shift: self.seed.unwrap().shift
                        + index
                            * self.glwe_size().0
                            * self.glwe_size.to_glwe_dimension().0
                            * self.polynomial_size().0
                            * <Self as AsRefTensor>::Element::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    seed,
                )
            })
    }

    /// Returns an iterator over mutably borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let mut ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for mut level_matrix in ggsw.level_matrix_iter_mut() {
    ///     for mut rlwe in level_matrix.row_iter_mut() {
    ///         rlwe.as_mut_tensor().fill_with_element(9);
    ///     }
    /// }
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<
        Item = GgswSeededLevelMatrix<
            &mut [<Self as AsMutTensor>::Element],
            <Self as AsRefTensor>::Element,
        >,
    >
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let matrix_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * glwe_size.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0
                            * <Self as AsMutTensor>::Element::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    matrix_seed,
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed level seeded matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// ggsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut rlwe in level_matrix.row_iter_mut() {
    ///             rlwe.as_mut_tensor().fill_with_element(9);
    ///         }
    ///     });
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<
        Item = GgswSeededLevelMatrix<
            &mut [<Self as AsRefTensor>::Element],
            <Self as AsRefTensor>::Element,
        >,
    >
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric + Sync + Send,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let matrix_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * glwe_size.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0
                            * <Self as AsMutTensor>::Element::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    matrix_seed,
                )
            })
    }

    /// Returns the ciphertext as a full fledged GgswCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::{
    ///     StandardGgswCiphertext, StandardGgswSeededCiphertext,
    /// };
    /// let seeded_ggsw = StandardGgswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 9 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut ggsw = StandardGgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// seeded_ggsw.expand_into(&mut ggsw);
    /// ```
    pub fn expand_into<Scalar, OutCont>(self, output: &mut StandardGgswCiphertext<OutCont>)
    where
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
        StandardGgswCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
    {
        let mut generator = RandomGenerator::new_from_seed(self.seed.unwrap());
        output
            .as_mut_glwe_list()
            .ciphertext_iter_mut()
            .zip(self.as_glwe_seeded_list().ciphertext_iter())
            .for_each(|(mut glwe_out, glwe_in)| {
                let (mut output_body, mut output_mask) = glwe_out.get_mut_body_and_mask();

                // generate a uniformly random mask
                generator.fill_tensor_with_random_uniform(output_mask.as_mut_tensor());

                output_body
                    .as_mut_tensor()
                    .as_mut_slice()
                    .clone_from_slice(glwe_in.into_tensor().as_slice());
            });
    }
}
