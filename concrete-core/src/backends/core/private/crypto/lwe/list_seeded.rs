use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{CiphertextCount, LweDimension, LweSize};
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::{
    tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::{LweList, LweSeededCiphertext};

/// A list of seeded ciphertexts encoded with the LWE scheme.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct LweSeededList<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) lwe_dimension: LweDimension,
    pub(crate) seed: u128,
    pub(crate) shift: usize,
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
    /// use concrete_core::crypto::{*, lwe::LweSeededList};
    /// let list = LweSeededList::allocate(0 as u8, LweSize(10), CiphertextCount(20));
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
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }
}

impl<Cont> LweSeededList<Cont> {
    /// Creates a list from a container, a lwe size and a seed.
    ///
    /// # Example:
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::LweSeededList};
    /// let list = LweSeededList::from_container(vec![0 as u8; 20], LweSize(10), 0);
    /// assert_eq!(list.count(), CiphertextCount(20));
    /// assert_eq!(list.lwe_size(), LweSize(10));
    /// ```
    pub fn from_container(cont: Cont, lwe_dimension: LweDimension, seed: u128, shift: usize) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        LweSeededList {
            tensor,
            lwe_dimension,
            seed,
            shift,
        }
    }

    /// Returns the number of ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, lwe::LweSeededList};
    /// let list = LweSeededList::from_container(vec![0 as u8; 20], LweSize(10), 0);
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
    /// use concrete_core::crypto::{*, lwe::LweSeededList};
    /// let list = LweSeededList::from_container(vec![0 as u8; 20], LweSize(10), 0);
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
    /// use concrete_core::crypto::{*, lwe::LweSeededList};
    /// let list = LweSeededList::from_container(vec![0 as u8; 20], LweSize(10), 0);
    /// assert_eq!(list.mask_size(), LweDimension(9));
    /// ```
    pub fn mask_size(&self) -> LweDimension {
        self.lwe_dimension
    }

    pub(crate) fn get_seed(&self) -> &u128 {
        &self.seed
    }

    /// Returns an iterator over seeded ciphertexts from the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, lwe::*};
    /// let list = LweSeededList::from_container(vec![0 as u8; 20], LweSize(10), 0);
    /// for ciphertext in list.ciphertext_iter(){
    ///     let (body, masks) = ciphertext.get_body_and_mask();
    ///     assert_eq!(body, &LweBody(0));
    ///     assert_eq!(masks, LweMask::from_container(&[0 as u8, 0, 0, 0, 0 ,0, 0, 0, 0][..]));
    /// }
    /// assert_eq!(list.ciphertext_iter.count(), 20);
    /// ```
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = LweSeededCiphertext<<Self as AsRefTensor>::Element>> + '_
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Copy,
    {
        self.as_tensor()
            .as_slice()
            .iter()
            .enumerate()
            .map(move |(i, b)| {
                LweSeededCiphertext::from_scalar(*b, self.lwe_dimension, self.seed, i)
            })
    }

    pub fn expand_into<OutCont, Scalar>(self, output: &mut LweList<OutCont>)
    where
        LweList<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
    {
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        for (mut lwe_out, body) in output
            .ciphertext_iter_mut()
            .zip(self.as_tensor().as_slice().iter())
        {
            let (output_body, mut output_mask) = lwe_out.get_mut_body_and_mask();

            // generate a uniformly random mask
            generator.fill_tensor_with_random_mask(output_mask.as_mut_tensor());
            output_body.0 = *body;
        }
    }
}
