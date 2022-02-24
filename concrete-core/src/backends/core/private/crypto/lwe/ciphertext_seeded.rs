#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{LweDimension, LweSize, Seed};

use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::AsMutTensor;

use super::{LweBody, LweCiphertext};

/// A seeded ciphertext encrypted using the LWE scheme.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweSeededCiphertext<Scalar> {
    pub(super) body: LweBody<Scalar>,
    pub(super) lwe_dimension: LweDimension,
    pub(super) seed: Option<Seed>,
}

impl<Scalar> LweSeededCiphertext<Scalar> {
    /// Allocates a new seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, LweSize};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededCiphertext;
    /// let ct = LweSeededCiphertext::allocate(0 as u8, LweDimension(3));
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn allocate(value: Scalar, lwe_dimension: LweDimension) -> Self {
        LweSeededCiphertext {
            body: LweBody(value),
            lwe_dimension,
            seed: None,
        }
    }

    /// Allocates a new seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededCiphertext;
    /// let ct = LweSeededCiphertext::from_scalar(0 as u8, LweDimension(3), Seed { seed: 0, shift: 0 });
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn from_scalar(value: Scalar, lwe_dimension: LweDimension, seed: Seed) -> Self {
        LweSeededCiphertext {
            body: LweBody(value),
            lwe_dimension,
            seed: Some(seed),
        }
    }

    /// Returns the size of the cipher, e.g. the size of the mask + 1 for the body.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, LweSize};
    /// use concrete_core::backends::core::private::crypto::lwe::LweSeededCiphertext;
    /// let ct = LweSeededCiphertext::allocate(0 as u8, LweDimension(3));
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_dimension.to_lwe_size()
    }

    /// Returns the seed of the ciphertext.
    pub(crate) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    /// Returns the seed of the ciphertext.
    pub(crate) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    /// Returns the body of the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::{LweBody, LweSeededCiphertext};
    /// let ciphertext =
    ///     LweSeededCiphertext::from_scalar(0 as u8, LweDimension(3), Seed { seed: 0, shift: 0 });
    /// let body = ciphertext.get_body();
    /// assert_eq!(body, &LweBody(0 as u8));
    /// ```
    pub fn get_body(&self) -> &LweBody<Scalar> {
        unsafe { &*{ &self.body.0 as *const Scalar as *const LweBody<Scalar> } }
    }

    /// Returns the mutable body of the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::{LweBody, LweSeededCiphertext};
    /// let mut ciphertext =
    ///     LweSeededCiphertext::from_scalar(0 as u8, LweDimension(3), Seed { seed: 0, shift: 0 });
    /// let mut body = ciphertext.get_mut_body();
    /// *body = LweBody(8);
    /// let body = ciphertext.get_body();
    /// assert_eq!(body, &LweBody(8 as u8));
    /// ```
    pub fn get_mut_body(&mut self) -> &mut LweBody<Scalar> {
        unsafe { &mut *{ &mut self.body.0 as *mut Scalar as *mut LweBody<Scalar> } }
    }

    /// Returns the ciphertext as a full fledged LweCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{LweDimension, LweSize, Seed};
    /// use concrete_core::backends::core::private::crypto::lwe::{
    ///     LweBody, LweCiphertext, LweSeededCiphertext,
    /// };
    /// let seeded_ciphertext =
    ///     LweSeededCiphertext::from_scalar(0 as u8, LweDimension(3), Seed { seed: 0, shift: 0 });
    /// let mut ciphertext = LweCiphertext::allocate(0 as u8, LweSize(10));
    /// seeded_ciphertext.expand_into(&mut ciphertext);
    /// let (body, mask) = ciphertext.get_mut_body_and_mask();
    /// assert_eq!(body, &mut LweBody(0));
    /// assert_eq!(mask.mask_size(), LweDimension(9));
    /// ```
    pub fn expand_into<Cont>(self, output: &mut LweCiphertext<Cont>)
    where
        LweCiphertext<Cont>: AsMutTensor<Element = Scalar>,
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
    {
        let mut generator = RandomGenerator::new_from_seed(self.seed.unwrap());
        let (output_body, mut output_mask) = output.get_mut_body_and_mask();

        // generate a uniformly random mask
        generator.fill_tensor_with_random_uniform(output_mask.as_mut_tensor());

        output_body.0 = self.body.0;
    }
}
