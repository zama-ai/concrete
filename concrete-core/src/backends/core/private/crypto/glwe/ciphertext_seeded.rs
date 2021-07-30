use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
use serde::{Deserialize, Serialize};

use crate::backends::core::private::{
    crypto::{bootstrap::FourierBuffers, secret::generators::EncryptionRandomGenerator},
    math::{
        fft::Complex64,
        random::RandomGenerator,
        tensor::{
            tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
        },
        torus::UnsignedTorus,
    },
};

use super::{GlweBody, GlweCiphertext};

/// An GLWE seeded ciphertext.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlweSeededCiphertext<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) glwe_dimension: GlweDimension,
    pub(crate) seed: u128,
    pub(crate) shift: usize,
}

tensor_traits!(GlweSeededCiphertext);

impl<Scalar> GlweSeededCiphertext<Vec<Scalar>> {
    /// Allocates a new GLWE seeded ciphertext, whose body and masks coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, glwe::*};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let glwe_ciphertext = GlweSeededCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100), 0_u128);
    /// assert_eq!(glwe_ciphertext.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(glwe_ciphertext.mask_size(), GlweDimension(99));
    /// assert_eq!(glwe_ciphertext.size(), GlweSize(100));
    /// ```
    pub fn allocate(value: Scalar, poly_size: PolynomialSize, dimension: GlweDimension) -> Self
    where
        Self: AsMutTensor,
        Scalar: Copy,
    {
        Self {
            tensor: Tensor::from_container(vec![value; poly_size.0]),
            glwe_dimension: dimension,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }
}

impl<Cont> GlweSeededCiphertext<Cont> {
    /// Creates a new GLWE seeded ciphertext from an existing container.
    ///
    /// # Note
    ///
    /// This method does not perform any transformation of the container data. Those are assumed to
    /// represent a valid glwe body.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, glwe::*};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let glwe = GlweSeededCiphertext::from_container(vec![0 as u8; 10], GlweDimension(109), 0_u128);
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(glwe.mask_size(), GlweDimension(109));
    /// assert_eq!(glwe.size(), GlweSize(110));
    /// ```
    pub fn from_container(cont: Cont, dimension: GlweDimension, seed: u128, shift: usize) -> Self {
        Self {
            tensor: Tensor::from_container(cont),
            glwe_dimension: dimension,
            seed,
            shift,
        }
    }

    /// Returns the size of the ciphertext, e.g. the number of masks + 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, glwe::*};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let glwe = GlweSeededCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100), 0_u128);
    /// assert_eq!(glwe.size(), GlweSize(100));
    /// ```
    pub fn size(&self) -> GlweSize {
        self.glwe_dimension.to_glwe_size()
    }

    /// Returns the number of masks of the ciphertext, e.g. its size - 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, glwe::*};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let glwe = GlweSeededCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100), 0_u128);
    /// assert_eq!(glwe.mask_size(), GlweDimension(99));
    /// ```
    pub fn mask_size(&self) -> GlweDimension {
        self.glwe_dimension
    }

    /// Returns the number of coefficients of the polynomials of the ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::{*, glwe::*};
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let glwe = GlweSeededCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100), 0_u128);
    /// assert_eq!(glwe_ciphertext.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize
    where
        Self: AsRefTensor,
    {
        PolynomialSize(self.as_tensor().len())
    }

    pub fn get_seed(&self) -> u128 {
        self.seed
    }

    pub fn get_body(&self) -> GlweBody<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        GlweBody {
            tensor: self.as_tensor().get_sub(0..),
        }
    }

    pub fn get_mut_body(&mut self) -> GlweBody<&mut [<Self as AsRefTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        GlweBody {
            tensor: self.as_mut_tensor().get_sub_mut(0..),
        }
    }

    /// Returns the ciphertext as a full fledged GlweCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::crypto::{*, lwe::*};
    /// let seeded_ciphertext = GlweSeededCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100), 0_u128);
    /// let mut ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// seeded_ciphertext.expand_into(&mut ciphertext);
    /// let (body, mask) = ciphertext.get_mut_body_and_mask();
    /// assert_eq!(body.as_polynomial().polynomial_size(), PolynomialSize(10));
    /// assert_eq!(mask.mask_element_iter().count(), 99);
    /// ```
    pub fn expand_into<Scalar, OutCont>(self, output: &mut GlweCiphertext<OutCont>)
    where
        Scalar: UnsignedTorus,
        GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
        Self: IntoTensor<Element = Scalar> + AsRefTensor,
    {
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        generator.shift(
            Scalar::BITS / 8 * self.glwe_dimension.0 * self.polynomial_size().0 * self.shift,
        );
        let (mut output_body, mut output_mask) = output.get_mut_body_and_mask();

        // generate a uniformly random mask
        generator.fill_tensor_with_random_mask(output_mask.as_mut_tensor());

        output_body
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(self.into_tensor().as_slice());
    }

    /// Returns the ciphertext as a full fledged GlweCiphertext in the FFT domain
    pub fn expand_into_complex<Scalar, OutCont>(
        self,
        output: &mut GlweCiphertext<OutCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        GlweCiphertext<OutCont>: AsMutTensor<Element = Complex64>,
        GlweCiphertext<Vec<Scalar>>: AsMutTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
        Self: IntoTensor<Element = Complex64> + AsRefTensor,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut buffers.fft_buffers.first_buffer;
        let fft = &mut buffers.fft_buffers.fft;

        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        generator.shift(
            Scalar::BITS / 8 * self.glwe_dimension.0 * self.polynomial_size().0 * self.shift,
        );

        let mut standard_ct =
            GlweCiphertext::allocate(Scalar::ZERO, output.poly_size, output.size());
        let mut standard_mask = standard_ct.get_mut_mask();

        // generate a uniformly random mask
        generator.fill_tensor_with_random_mask(standard_mask.as_mut_tensor());

        // Converts into FFT domain
        for (mut poly_fft, poly_standard) in output
            .get_mut_mask()
            .as_mut_polynomial_list()
            .polynomial_iter_mut()
            .zip(standard_mask.as_polynomial_list().polynomial_iter())
        {
            fft.forward_as_torus(fft_buffer, &poly_standard);
            poly_fft
                .as_mut_tensor()
                .fill_with_one(fft_buffer.as_tensor(), |a| *a);
        }

        output
            .get_mut_body()
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(&self.into_tensor().as_slice());
    }
}
