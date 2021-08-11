use crate::math::polynomial::Polynomial;
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor};
use crate::tensor_traits;

/// The body of a GLWE ciphertext.
pub struct GlweBody<Cont> {
    pub(super) tensor: Tensor<Cont>,
}

tensor_traits!(GlweBody);

impl<Cont> GlweBody<Cont> {
    /// Consumes the current ciphertext body, and return a polynomial over the original container.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::*;
    /// use concrete_core::crypto::*;
    /// let glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let body = glwe.get_body();
    /// let poly = body.into_polynomial();
    /// assert_eq!(poly.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn into_polynomial(self) -> Polynomial<Cont>
    where
        Self: IntoTensor<Container = Cont>,
    {
        Polynomial::from_container(self.into_tensor().into_container())
    }

    /// Returns a borrowed polynomial from the current body.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::*;
    /// use concrete_core::crypto::*;
    /// let glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let body = glwe.get_body();
    /// let poly = body.as_polynomial();
    /// assert_eq!(poly.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn as_polynomial(&self) -> Polynomial<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        Polynomial::from_container(self.as_tensor().as_slice())
    }

    /// Returns a mutably borrowed polynomial from the current body.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::{GlweSize, PolynomialSize};
    /// use concrete_core::crypto::glwe::*;
    /// use concrete_core::crypto::*;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut glwe = GlweCiphertext::allocate(0 as u8, PolynomialSize(10), GlweSize(100));
    /// let mut body = glwe.get_mut_body();
    /// let mut poly = body.as_mut_polynomial();
    /// poly.as_mut_tensor().fill_with_element(9);
    /// assert!(body.as_tensor().iter().all(|a| *a == 9));
    /// ```
    pub fn as_mut_polynomial(&mut self) -> Polynomial<&mut [<Self as AsMutTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        Polynomial::from_container(self.as_mut_tensor().as_mut_slice())
    }
}
