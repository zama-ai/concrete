#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize, Seed};

use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::{
    tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::{LweList, LweSeededCiphertext};

/// A list of seeded ciphertexts encoded with the LWE scheme.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweSeededList<Cont> {
    tensor: Tensor<Cont>,
    lwe_dimension: LweDimension,
    seed: Option<Seed>,
}

tensor_traits!(LweSeededList);

impl<Scalar> LweSeededList<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates a list of seeded lwe ciphertexts whose all bodies have the value `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededList;
    /// let list = LweSeededList::allocate(0 as u8, LweDimension(9), CiphertextCount(20));
    /// assert_eq!(list.count(), CiphertextCount(20));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn allocate(
        value: Scalar,
        lwe_dimension: LweDimension,
        lwe_count: CiphertextCount,
    ) -> Self {
        LweSeededList {
            tensor: Tensor::from_container(vec![value; lwe_count.0]),
            lwe_dimension,
            seed: None,
        }
    }
}

impl<Cont> LweSeededList<Cont> {
    /// Creates a list from a container, a lwe dimension and a seed.
    ///
    /// # Example:
    ///
    /// ```
    /// use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededList;
    /// let list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(list.count(), CiphertextCount(20));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn from_container(cont: Cont, lwe_dimension: LweDimension, seed: Seed) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        LweSeededList {
            tensor,
            lwe_dimension,
            seed: Some(seed),
        }
    }

    /// Returns the number of ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededList;
    /// let list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(list.count(), CiphertextCount(20));
    /// ```
    pub fn count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        CiphertextCount(self.as_tensor().len())
    }

    /// Returns the size of the ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededList;
    /// let list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_dimension.to_lwe_size()
    }

    /// Returns the number of masks of the ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{LweDimension, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededList;
    /// let list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(list.mask_size(), LweDimension(9));
    /// ```
    pub fn mask_size(&self) -> LweDimension {
        self.lwe_dimension
    }

    pub(crate) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    pub(crate) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    /// Returns an iterator over seeded ciphertexts from the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{LweDimension, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::{LweBody, LweSeededList};
    /// let list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for ciphertext in list.ciphertext_iter() {
    ///     let body = ciphertext.get_body();
    ///     assert_eq!(body, &LweBody(0));
    /// }
    /// assert_eq!(list.ciphertext_iter().count(), 20);
    /// ```
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = LweSeededCiphertext<<Self as AsRefTensor>::Element>> + '_
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        self.as_tensor()
            .as_slice()
            .iter()
            .enumerate()
            .map(move |(i, b)| {
                let seed = Seed {
                    seed: self.seed.unwrap().seed,
                    shift: self.seed.unwrap().shift
                        + <Self as AsRefTensor>::Element::BITS / 8 * self.lwe_dimension.0 * i,
                };
                LweSeededCiphertext::from_scalar(*b, self.lwe_dimension, seed)
            })
    }

    /// Returns the ciphertext list as a full fledged LweList
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::{LweBody, LweList, LweSeededList};
    /// let seeded_list = LweSeededList::from_container(
    ///     vec![0 as u8; 20],
    ///     LweDimension(9),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut list = LweList::allocate(0 as u8, LweSize(10), CiphertextCount(20));
    /// seeded_list.expand_into(&mut list);
    /// assert_eq!(list.mask_size(), LweDimension(9));
    /// ```
    pub fn expand_into<OutCont, Scalar>(self, output: &mut LweList<OutCont>)
    where
        LweList<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
    {
        let mut generator = RandomGenerator::new_from_seed(self.seed.unwrap());
        for (mut lwe_out, lwe_in) in output.ciphertext_iter_mut().zip(self.ciphertext_iter()) {
            let (output_body, mut output_mask) = lwe_out.get_mut_body_and_mask();

            // generate a uniformly random mask
            generator.fill_tensor_with_random_uniform(output_mask.as_mut_tensor());
            output_body.0 = lwe_in.get_body().0;
        }
    }
}
