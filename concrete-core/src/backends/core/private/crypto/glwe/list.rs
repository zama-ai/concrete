#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::GlweCiphertext;
use concrete_commons::parameters::{CiphertextCount, GlweDimension, GlweSize, PolynomialSize};

/// A list of ciphertexts encoded with the GLWE scheme.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GlweList<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) rlwe_size: GlweSize,
    pub(crate) poly_size: PolynomialSize,
}

tensor_traits!(GlweList);

impl<Scalar> GlweList<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates storage for an owned [`GlweList`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
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
        GlweList {
            tensor: Tensor::from_container(vec![
                value;
                poly_size.0
                    * (glwe_dimension.0 + 1)
                    * ciphertext_number.0
            ]),
            rlwe_size: GlweSize(glwe_dimension.0 + 1),
            poly_size,
        }
    }
}

impl<Cont> GlweList<Cont> {
    /// Creates a list from a container of values.
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::from_container(
    ///     vec![0 as u8; 10 * 21 * 30],
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
        rlwe_dimension: GlweDimension,
        poly_size: PolynomialSize,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => rlwe_dimension.0 + 1, poly_size.0);
        GlweList {
            tensor,
            rlwe_size: GlweSize(rlwe_dimension.0 + 1),
            poly_size,
        }
    }

    /// Returns the number of ciphertexts in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    /// );
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(30));
    /// ```
    pub fn ciphertext_count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.rlwe_size.0, self.poly_size.0);
        CiphertextCount(self.as_tensor().len() / (self.rlwe_size.0 * self.polynomial_size().0))
    }

    /// Returns the size of the glwe ciphertexts contained in the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    /// );
    /// assert_eq!(list.glwe_size(), GlweSize(21));
    /// ```
    pub fn glwe_size(&self) -> GlweSize
    where
        Self: AsRefTensor,
    {
        self.rlwe_size
    }

    /// Returns the number of coefficients of the polynomials used for the list ciphertexts.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
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
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::GlweList;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    /// );
    /// assert_eq!(list.glwe_dimension(), GlweDimension(20));
    /// ```
    pub fn glwe_dimension(&self) -> GlweDimension
    where
        Self: AsRefTensor,
    {
        GlweDimension(self.rlwe_size.0 - 1)
    }

    /// Returns an iterator over ciphertexts borrowed from the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::{GlweBody, GlweList};
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    /// let list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    /// );
    /// for ciphertext in list.ciphertext_iter() {
    ///     let (body, masks) = ciphertext.get_body_and_mask();
    ///     assert_eq!(body.as_polynomial().polynomial_size(), PolynomialSize(10));
    /// }
    /// assert_eq!(list.ciphertext_iter().count(), 30);
    /// ```
    pub fn ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = GlweCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.rlwe_size.0, self.poly_size.0);
        let poly_size = self.poly_size;
        let size = self.rlwe_size.0 * self.polynomial_size().0;
        self.as_tensor()
            .subtensor_iter(size)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), poly_size))
    }

    /// Returns an iterator over ciphertexts borrowed from the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
    /// use concrete_core::backends::core::private::crypto::glwe::{GlweBody, GlweList};
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut list = GlweList::allocate(
    ///     0 as u8,
    ///     PolynomialSize(10),
    ///     GlweDimension(20),
    ///     CiphertextCount(30),
    /// );
    /// for mut ciphertext in list.ciphertext_iter_mut() {
    ///     let mut body = ciphertext.get_mut_body();
    ///     body.as_mut_tensor().fill_with_element(9);
    /// }
    /// for ciphertext in list.ciphertext_iter() {
    ///     let body = ciphertext.get_body();
    ///     assert!(body.as_tensor().iter().all(|a| *a == 9));
    /// }
    /// assert_eq!(list.ciphertext_iter_mut().count(), 30);
    /// ```
    pub fn ciphertext_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = GlweCiphertext<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        ck_dim_div!(self.as_tensor().len() => self.rlwe_size.0, self.poly_size.0);
        let poly_size = self.poly_size;
        let chunks_size = self.rlwe_size.0 * self.polynomial_size().0;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |sub| GlweCiphertext::from_container(sub.into_container(), poly_size))
    }
}
