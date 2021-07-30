use concrete_commons::parameters::{CiphertextCount, GlweDimension, GlweSize, PolynomialSize};
use serde::{Deserialize, Serialize};

use crate::backends::core::private::{
    crypto::secret::generators::EncryptionRandomGenerator,
    math::{
        random::{RandomGenerable, RandomGenerator, Uniform},
        tensor::{
            ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
        },
    },
};

use super::{GlweList, GlweSeededCiphertext};

/// A list of ciphertexts encoded with the GLWE scheme.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlweSeededList<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) glwe_dimension: GlweDimension,
    pub(crate) poly_size: PolynomialSize,
    pub(crate) seed: u128,
    pub(crate) shift: usize,
}

tensor_traits!(GlweSeededList);

impl<Scalar> GlweSeededList<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates storage for an owned [`GlweSeededList`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweSize, CiphertextCount, GlweDimension};
    /// let list = GlweSeededList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    ///     0 as u128,
    /// );
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(30));
    /// assert_eq!(list.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(list.glwe_size(), GlweSize(21));
    /// assert_eq!(list.glwe_dimension(), GlweDimension(20));
    /// ```
    pub fn allocate(
        value: Scalar,
        poly_size: PolynomialSize,
        glwe_dimension: GlweDimension,
        ciphertext_number: CiphertextCount,
    ) -> Self {
        GlweSeededList {
            tensor: Tensor::from_container(vec![value; poly_size.0 * ciphertext_number.0]),
            glwe_dimension,
            poly_size,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }
}

impl<Cont> GlweSeededList<Cont> {
    /// Creates a list from a container of values.
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweSize, CiphertextCount, GlweDimension};
    /// let list = GlweSeededList::from_container(
    ///     vec![0 as u8; 10 * 30],
    ///     GlweDimension(20),
    ///     PolynomialSize(10),
    /// );
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(30));
    /// assert_eq!(list.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(list.glwe_size(), GlweSize(21));
    /// assert_eq!(list.glwe_dimension(), GlweDimension(20));
    /// ```
    pub fn from_container(
        cont: Cont,
        glwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
        seed: u128,
        shift: usize,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => poly_size.0);
        GlweSeededList {
            tensor,
            glwe_dimension,
            poly_size,
            seed,
            shift,
        }
    }

    /// Returns the number of ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweSize, CiphertextCount, GlweDimension};
    /// let list = GlweSeededList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    ///     0 as u128,
    /// );
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(30));
    /// ```
    pub fn ciphertext_count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.poly_size.0);
        CiphertextCount(self.as_tensor().len() / self.polynomial_size().0)
    }

    /// Returns the size of the glwe ciphertexts contained in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweDimension, CiphertextCount, GlweSize};
    /// let list = GlweSeededList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    ///     0 as u128,
    /// );
    /// assert_eq!(list.glwe_size(), GlweSize(21));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_dimension.to_glwe_size()
    }

    /// Returns the number of coefficients of the polynomials used for the list ciphertexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweSize, CiphertextCount, GlweDimension};
    /// let list = GlweSeededList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    ///     0 as u128,
    /// );
    /// assert_eq!(list.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns the number of masks of the ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_core::crypto::glwe::GlweSeededList;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// use concrete_core::crypto::{GlweSize, CiphertextCount, GlweDimension};
    /// let list = GlweSeededList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    ///     0 as u128,
    /// );
    /// assert_eq!(list.glwe_dimension(), GlweDimension(20));
    /// ```
    pub fn glwe_dimension(&self) -> GlweDimension {
        self.glwe_dimension
    }

    pub fn get_seed(&self) -> u128 {
        self.seed
    }

    /// Returns an iterator over ciphertexts expanded from the list.
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = GlweSeededCiphertext<Vec<<Self as AsRefTensor>::Element>>> + '_
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Clone,
    {
        self.as_tensor()
            .as_slice()
            .chunks(self.poly_size.0)
            .enumerate()
            .map(move |(i, body)| {
                GlweSeededCiphertext::from_container(
                    body.to_vec(),
                    self.glwe_dimension(),
                    self.seed,
                    i,
                )
            })
    }

    pub fn expand_into<OutCont, Scalar>(self, output: &mut GlweList<OutCont>)
    where
        Self: AsRefTensor<Element = Scalar>,
        GlweList<OutCont>: AsMutTensor<Element = Scalar>,
        Scalar: Clone + RandomGenerable<Uniform>,
    {
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));

        for (mut glwe_out, body_in) in output.ciphertext_iter_mut().zip(self.ciphertext_iter()) {
            let (mut body, mut mask) = glwe_out.get_mut_body_and_mask();
            generator.fill_tensor_with_random_mask(mask.as_mut_tensor());
            body.as_mut_tensor()
                .as_mut_slice()
                .clone_from_slice(body_in.as_tensor().as_slice());
        }
    }
}
