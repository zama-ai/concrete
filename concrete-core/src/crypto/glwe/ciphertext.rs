use serde::{Deserialize, Serialize};

use crate::math::polynomial::PolynomialList;
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use crate::tensor_traits;

use super::{GlweBody, GlweMask};
use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};

/// An GLWE ciphertext.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlweCiphertext<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) poly_size: PolynomialSize,
}

tensor_traits!(GlweCiphertext);

impl<Scalar> GlweCiphertext<Vec<Scalar>> {
    /// Allocates a new GLWE ciphertext, whose body and masks coefficients are
    /// all `value`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let glwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// assert_eq!(glwe_ciphertext.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(glwe_ciphertext.mask_size(), GlweDimension(99));
    /// assert_eq!(glwe_ciphertext.size(), GlweSize(100));
    /// ```
    pub fn allocate(
        value: Scalar,
        poly_size: PolynomialSize,
        size: GlweSize,
    ) -> GlweCiphertext<Vec<Scalar>>
    where
        GlweCiphertext<Vec<Scalar>>: AsMutTensor,
        Scalar: Copy,
    {
        GlweCiphertext::from_container(vec![value; poly_size.0 * size.0], poly_size)
    }
}

impl<Cont> GlweCiphertext<Cont> {
    /// Creates a new GLWE ciphertext from an existing container.
    ///
    /// # Note
    ///
    /// This method does not perform any transformation of the container data.
    /// Those are assumed to represent a valid glwe ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let glwe = GlweCiphertext::from_container(vec![0 as u8; 1100], PolynomialSize(10));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(glwe.mask_size(), GlweDimension(109));
    /// assert_eq!(glwe.size(), GlweSize(110));
    /// ```
    pub fn from_container(cont: Cont, poly_size: PolynomialSize) -> GlweCiphertext<Cont> {
        GlweCiphertext {
            tensor: Tensor::from_container(cont),
            poly_size,
        }
    }

    /// Returns the size of the ciphertext, e.g. the number of masks + 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// assert_eq!(glwe.size(), GlweSize(100));
    /// ```
    pub fn size(&self) -> GlweSize
    where
        Self: AsRefTensor,
    {
        GlweSize(self.as_tensor().len() / self.poly_size.0)
    }

    /// Returns the number of masks of the ciphertext, e.g. its size - 1.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// assert_eq!(glwe.mask_size(), GlweDimension(99));
    /// ```
    pub fn mask_size(&self) -> GlweDimension
    where
        Self: AsRefTensor,
    {
        GlweDimension(self.size().0 - 1)
    }

    /// Returns the number of coefficients of the polynomials of the ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let rlwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// assert_eq!(rlwe_ciphertext.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed [`GlweBody`] from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let rlwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let body = rlwe_ciphertext.get_body();
    /// assert_eq!(body.as_polynomial().polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn get_body(&self) -> GlweBody<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        GlweBody {
            tensor: self
                .as_tensor()
                .get_sub((self.mask_size().0 * self.polynomial_size().0)..),
        }
    }

    /// Returns a borrowed [`GlweMask`] from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let rlwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let mask = rlwe_ciphertext.get_mask();
    /// assert_eq!(mask.mask_element_iter().count(), 99);
    /// ```
    pub fn get_mask(&self) -> GlweMask<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        GlweMask {
            tensor: self
                .as_tensor()
                .get_sub(..(self.mask_size().0 * self.polynomial_size().0)),
            poly_size: self.poly_size,
        }
    }

    /// Returns a mutably borrowed [`GlweBody`] from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let mut body = glwe.get_mut_body();
    /// body.as_mut_tensor().fill_with_element(9);
    /// let body = glwe.get_body();
    /// assert!(body.as_tensor().iter().all(|a| *a == 9));
    /// ```
    pub fn get_mut_body(&mut self) -> GlweBody<&mut [<Self as AsRefTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        let body_index = self.mask_size().0 * self.polynomial_size().0;
        GlweBody {
            tensor: self.as_mut_tensor().get_sub_mut(body_index..),
        }
    }

    /// Returns a mutably borrowed [`GlweMask`] from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let mut masks = glwe.get_mut_mask();
    /// for mut mask in masks.mask_element_iter_mut() {
    ///     mask.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert_eq!(masks.mask_element_iter_mut().count(), 99);
    /// assert!(!glwe.as_tensor().iter().all(|a| *a == 9));
    /// ```
    pub fn get_mut_mask(&mut self) -> GlweMask<&mut [<Self as AsRefTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        let body_index = self.mask_size().0 * self.polynomial_size().0;
        let poly_size = self.poly_size;
        GlweMask {
            tensor: self.as_mut_tensor().get_sub_mut(..body_index),
            poly_size,
        }
    }

    /// Returns borrowed [`GlweBody`] and [`GlweMask`] from the current
    /// ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let (body, masks) = glwe.get_body_and_mask();
    /// assert_eq!(body.as_polynomial().polynomial_size(), PolynomialSize(10));
    /// assert_eq!(masks.mask_element_iter().count(), 99);
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn get_body_and_mask(
        &self,
    ) -> (
        GlweBody<&[<Self as AsRefTensor>::Element]>,
        GlweMask<&[<Self as AsRefTensor>::Element]>,
    )
    where
        Self: AsRefTensor,
    {
        let index = self.mask_size().0 * self.polynomial_size().0;
        (
            GlweBody {
                tensor: self.as_tensor().get_sub(index..),
            },
            GlweMask {
                tensor: self.as_tensor().get_sub(..index),
                poly_size: self.poly_size,
            },
        )
    }

    /// Returns borrowed [`GlweBody`] and [`GlweMask`] from the current
    /// ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let (mut body, mut masks) = glwe.get_mut_body_and_mask();
    /// body.as_mut_tensor().fill_with_element(9);
    /// for mut mask in masks.mask_element_iter_mut() {
    ///     mask.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert_eq!(body.as_polynomial().polynomial_size(), PolynomialSize(10));
    /// assert!(glwe.as_tensor().iter().all(|a| *a == 9));
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn get_mut_body_and_mask(
        &mut self,
    ) -> (
        GlweBody<&mut [<Self as AsRefTensor>::Element]>,
        GlweMask<&mut [<Self as AsRefTensor>::Element]>,
    )
    where
        Self: AsMutTensor,
    {
        let body_index = self.mask_size().0 * self.polynomial_size().0;
        let poly_size = self.poly_size;
        let (masks, body) = self.as_mut_tensor().as_mut_slice().split_at_mut(body_index);
        (
            GlweBody {
                tensor: Tensor::from_container(body),
            },
            GlweMask {
                tensor: Tensor::from_container(masks),
                poly_size,
            },
        )
    }

    /// Consumes the current ciphertext and turn it to a list of polynomial.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialCount, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let rlwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let poly_list = rlwe_ciphertext.into_polynomial_list();
    /// assert_eq!(poly_list.polynomial_count(), PolynomialCount(100));
    /// assert_eq!(poly_list.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn into_polynomial_list(self) -> PolynomialList<Cont> {
        PolynomialList {
            tensor: self.tensor,
            poly_size: self.poly_size,
        }
    }

    /// Returns a borrowed polynomial list from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialCount, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// let rlwe_ciphertext = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let poly_list = rlwe_ciphertext.as_polynomial_list();
    /// assert_eq!(poly_list.polynomial_count(), PolynomialCount(100));
    /// assert_eq!(poly_list.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn as_polynomial_list(&self) -> PolynomialList<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        PolynomialList {
            tensor: Tensor::from_container(self.as_tensor().as_slice()),
            poly_size: self.poly_size,
        }
    }

    /// Returns a mutably borrowed polynomial list from the current ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let mut poly_list = glwe.as_mut_polynomial_list();
    /// for mut poly in poly_list.polynomial_iter_mut() {
    ///     poly.as_mut_tensor().fill_with_element(9);
    /// }
    /// assert!(glwe.as_tensor().iter().all(|a| *a == 9));
    /// ```
    pub fn as_mut_polynomial_list(
        &mut self,
    ) -> PolynomialList<&mut [<Self as AsMutTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        let poly_size = self.poly_size;
        PolynomialList {
            tensor: Tensor::from_container(self.as_mut_tensor().as_mut_slice()),
            poly_size,
        }
    }
}
