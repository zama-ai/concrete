use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{LweDimension, LweSize};
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, Uniform};
use crate::backends::core::private::math::tensor::AsMutTensor;

use super::{LweBody, LweCiphertext};

/// A seeded ciphertext encrypted using the LWE scheme.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LweSeededCiphertext<Scalar> {
    pub(super) body: LweBody<Scalar>,
    pub(super) lwe_dimension: LweDimension,
    pub(super) seed: u128,
    pub(super) shift: usize,
}

impl<Scalar> LweSeededCiphertext<Scalar> {
    /// Allocates a new seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::LweSeededCiphertext};
    /// let ct = LweSeededCiphertext::allocate(0 as u8, LWEDimension(3));
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn allocate(value: Scalar, lwe_dimension: LweDimension) -> Self {
        LweSeededCiphertext {
            body: LweBody(value),
            lwe_dimension,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }

    /// Allocates a new seeded ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::LweSeededCiphertext};
    /// let ct = LweSeededCiphertext::from_scalar(0 as u8, LWEDimension(3), 0);
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn from_scalar(
        value: Scalar,
        lwe_dimension: LweDimension,
        seed: u128,
        shift: usize,
    ) -> Self {
        LweSeededCiphertext {
            body: LweBody(value),
            lwe_dimension,
            seed,
            shift,
        }
    }

    /// Returns the size of the cipher, e.g. the size of the mask + 1 for the body.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::LweSeededCiphertext};
    /// let ct = LweSeededCiphertext::allocate(0 as u8, LweDimension(3), 0);
    /// assert_eq!(ct.lwe_size(), LweSize(4));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_dimension.to_lwe_size()
    }

    pub(crate) fn get_seed(&self) -> u128 {
        self.seed
    }

    pub(crate) fn get_shift(&self) -> usize {
        self.shift
    }

    #[allow(dead_code)]
    pub(crate) fn get_body(&self) -> &LweBody<Scalar> {
        unsafe { &*{ &self.body.0 as *const Scalar as *const LweBody<Scalar> } }
    }

    pub(crate) fn get_mut_body(&mut self) -> &mut LweBody<Scalar> {
        unsafe { &mut *{ &mut self.body.0 as *mut Scalar as *mut LweBody<Scalar> } }
    }

    /// Returns the ciphertext as a full fledged LweCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::*};
    /// let seeded_ciphertext = LweSeededCiphertext::allocate(0 as u8, LweDimension(9), 0);
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
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        generator.shift(Scalar::BITS / 8 * self.lwe_dimension.0 * self.shift);
        let (output_body, mut output_mask) = output.get_mut_body_and_mask();

        // generate a uniformly random mask
        generator.fill_tensor_with_random_mask(output_mask.as_mut_tensor());

        output_body.0 = self.body.0;
    }
}
