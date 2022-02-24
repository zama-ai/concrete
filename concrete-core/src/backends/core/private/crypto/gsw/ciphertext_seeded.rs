use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::lwe::LweSeededList;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::{GswCiphertext, GswSeededLevelMatrix};

/// A GSW seeded ciphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(PartialEq, Debug, Clone)]
pub struct GswSeededCiphertext<Cont> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    decomp_base_log: DecompositionBaseLog,
    seed: Option<Seed>,
}

tensor_traits!(GswSeededCiphertext);

impl<Scalar> GswSeededCiphertext<Vec<Scalar>> {
    /// Allocates a new GSW seeded ciphertext whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn allocate(
        value: Scalar,
        lwe_size: LweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Numeric,
    {
        GswSeededCiphertext {
            tensor: Tensor::from_container(vec![value; decomp_level.0 * lwe_size.0]),
            lwe_size,
            decomp_base_log,
            seed: None,
        }
    }
}

impl<Cont> GswSeededCiphertext<Cont> {
    /// Creates a gsw ciphertext from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        decomp_base_log: DecompositionBaseLog,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice,
        <Cont as AsRefSlice>::Element: Numeric,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => lwe_size.0);
        GswSeededCiphertext {
            tensor,
            lwe_size,
            decomp_base_log,
            seed: Some(seed),
        }
    }

    /// Returns the size of the lwe ciphertexts composing the gsw seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    pub(crate) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    pub(crate) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    /// Returns the number of decomposition levels used in the seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.lwe_size.0
        );
        DecompositionLevelCount(self.as_tensor().len() / self.lwe_size.0)
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns a borrowed list composed of all the LWE seeded ciphertexts composing current
    /// ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let list = gsw.as_lwe_list();
    /// assert_eq!(list.lwe_size(), LweSize(7));
    /// assert_eq!(list.count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_lwe_list<Scalar>(&self) -> LweSeededList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        LweSeededList::from_container(
            self.as_tensor().as_slice(),
            self.lwe_size.to_lwe_dimension(),
            self.seed.unwrap(),
        )
    }

    /// Returns a mutably borrowed `LweSeededList` composed of all the LWE seeded ciphertexts
    /// composing current ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut list = gsw.as_mut_lwe_list();
    /// list.as_mut_tensor().fill_with_element(0);
    /// assert_eq!(list.lwe_size(), LweSize(7));
    /// assert_eq!(list.count(), CiphertextCount(3 * 7));
    /// gsw.as_tensor().iter().for_each(|a| assert_eq!(*a, 0));
    /// ```
    pub fn as_mut_lwe_list<Scalar>(&mut self) -> LweSeededList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let lwe_dimension = self.lwe_size.to_lwe_dimension();
        let seed = self.seed.unwrap();
        LweSeededList::from_container(self.as_mut_tensor().as_mut_slice(), lwe_dimension, seed)
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
    ///     DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// let gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for level_matrix in gsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     for lwe in level_matrix.row_iter() {
    ///         assert_eq!(lwe.lwe_size(), LweSize(7));
    ///     }
    /// }
    /// assert_eq!(gsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GswSeededLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        let lwe_size = self.lwe_size;
        let seed = self.seed.unwrap();
        self.as_tensor()
            .subtensor_iter(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                let level_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * lwe_size.0
                            * lwe_size.to_lwe_dimension().0
                            * <Self as AsRefTensor>::Element::BITS
                            / 8,
                };
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    level_seed,
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
    ///     DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for mut level_matrix in gsw.level_matrix_iter_mut() {
    ///     for mut lwe in level_matrix.row_iter_mut() {
    ///         **lwe.as_mut_scalar() = 9;
    ///     }
    /// }
    /// assert!(gsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(gsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        let lwe_size = self.lwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .subtensor_iter_mut(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                let level_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * lwe_size.0
                            * lwe_size.to_lwe_dimension().0
                            * <Self as AsMutTensor>::Element::BITS
                            / 8,
                };
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    level_seed,
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed seeded level matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    /// let mut gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// gsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut lwe in level_matrix.row_iter_mut() {
    ///             **lwe.as_mut_scalar() = 9;
    ///         }
    ///     });
    /// assert!(gsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(gsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric + Sync + Send,
    {
        let lwe_size = self.lwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .par_subtensor_iter_mut(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                let level_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * lwe_size.0
                            * lwe_size.to_lwe_dimension().0
                            * <Self as AsMutTensor>::Element::BITS
                            / 8,
                };
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    level_seed,
                )
            })
    }

    /// Returns the ciphertext as a full fledged GswCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::{GswCiphertext, GswSeededCiphertext};
    /// let seeded_gsw = GswSeededCiphertext::from_container(
    ///     vec![9 as u8; 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut gsw = GswCiphertext::allocate(
    ///     0 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// seeded_gsw.expand_into(&mut gsw);
    /// ```
    pub fn expand_into<OutCont, Scalar>(self, output: &mut GswCiphertext<OutCont, Scalar>)
    where
        GswCiphertext<OutCont, Scalar>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: Numeric + RandomGenerable<Uniform>,
    {
        let mut generator = RandomGenerator::new_from_seed(self.seed.unwrap());
        for (mut lwe_out, body_in) in output
            .as_mut_lwe_list()
            .ciphertext_iter_mut()
            .zip(self.as_lwe_list().ciphertext_iter())
        {
            let (mut body, mut mask) = lwe_out.get_mut_body_and_mask();
            generator.fill_tensor_with_random_uniform(mask.as_mut_tensor());
            body.0 = body_in.get_body().0;
        }
    }
}
