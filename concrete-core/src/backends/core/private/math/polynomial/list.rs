use std::iter::Iterator;

use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::*;
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{MonomialDegree, PolynomialCount, PolynomialSize};

/// A generic polynomial list type.
///
/// This type represents a set of polynomial of homogeneous degree.
///
/// # Example
///
/// ```
/// use concrete_commons::parameters::{PolynomialCount, PolynomialSize};
/// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
/// let list = PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
/// assert_eq!(list.polynomial_count(), PolynomialCount(4));
/// assert_eq!(list.polynomial_size(), PolynomialSize(2));
/// ```
#[derive(PartialEq)]
pub struct PolynomialList<Cont> {
    pub(crate) tensor: Tensor<Cont>,
    pub(crate) poly_size: PolynomialSize,
}

tensor_traits!(PolynomialList);

impl<Coef> PolynomialList<Vec<Coef>>
where
    Coef: Copy,
{
    /// Allocates a new polynomial list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{PolynomialCount, PolynomialSize};
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let list = PolynomialList::allocate(1u8, PolynomialCount(10), PolynomialSize(2));
    /// assert_eq!(list.polynomial_count(), PolynomialCount(10));
    /// assert_eq!(list.polynomial_size(), PolynomialSize(2));
    /// ```
    pub fn allocate(value: Coef, number: PolynomialCount, size: PolynomialSize) -> Self {
        PolynomialList {
            tensor: Tensor::from_container(vec![value; number.0 * size.0]),
            poly_size: size,
        }
    }
}

impl<Cont> PolynomialList<Cont> {
    /// Creates a polynomial list from a list of values.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{PolynomialCount, PolynomialSize};
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let list = PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// assert_eq!(list.polynomial_count(), PolynomialCount(4));
    /// assert_eq!(list.polynomial_size(), PolynomialSize(2));
    /// ```
    pub fn from_container(cont: Cont, poly_size: PolynomialSize) -> PolynomialList<Cont>
    where
        Cont: AsRefSlice,
    {
        ck_dim_div!(cont.as_slice().len() => poly_size.0);
        PolynomialList {
            tensor: Tensor::from_container(cont),
            poly_size,
        }
    }

    /// Returns the number of polynomials in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{PolynomialCount, PolynomialSize};
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let list = PolynomialList::allocate(1u8, PolynomialCount(10), PolynomialSize(2));
    /// assert_eq!(list.polynomial_count(), PolynomialCount(10));
    /// ```
    pub fn polynomial_count(&self) -> PolynomialCount
    where
        Self: AsRefTensor,
    {
        PolynomialCount(self.as_tensor().len() / self.poly_size.0)
    }

    /// Returns the size of the polynomials in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{PolynomialCount, PolynomialSize};
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let list = PolynomialList::allocate(1u8, PolynomialCount(10), PolynomialSize(2));
    /// assert_eq!(list.polynomial_size(), PolynomialSize(2));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a reference to the n-th polynomial of the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let list = PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// let poly = list.get_polynomial(2);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(0)).get_coefficient(), 5u8);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(1)).get_coefficient(), 6u8);
    /// ```
    pub fn get_polynomial(&self, n: usize) -> Polynomial<&[<Self as AsRefTensor>::Element]>
    where
        Self: AsRefTensor,
    {
        Polynomial {
            tensor: self
                .as_tensor()
                .get_sub((n * self.poly_size.0)..(n + 1) * self.poly_size.0),
        }
    }

    /// Returns a mutable reference to the n-th polynomial of the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// let mut poly = list.get_mut_polynomial(2);
    /// poly.get_mut_monomial(MonomialDegree(0))
    ///     .set_coefficient(10u8);
    /// poly.get_mut_monomial(MonomialDegree(1))
    ///     .set_coefficient(11u8);
    /// let poly = list.get_polynomial(2);
    /// assert_eq!(
    ///     *poly.get_monomial(MonomialDegree(0)).get_coefficient(),
    ///     10u8
    /// );
    /// assert_eq!(
    ///     *poly.get_monomial(MonomialDegree(1)).get_coefficient(),
    ///     11u8
    /// );
    /// ```
    pub fn get_mut_polynomial(
        &mut self,
        n: usize,
    ) -> Polynomial<&mut [<Self as AsRefTensor>::Element]>
    where
        Self: AsMutTensor,
    {
        let index = (n * self.poly_size.0)..((n + 1) * self.poly_size.0);
        Polynomial {
            tensor: self.as_mut_tensor().get_sub_mut(index),
        }
    }

    /// Returns an iterator over references to the polynomials contained in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::PolynomialList;
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// assert_eq!(list.polynomial_iter().count(), 4);
    /// ```
    pub fn polynomial_iter(
        &self,
    ) -> impl Iterator<Item = Polynomial<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .map(|sub| Polynomial::from_container(sub.into_container()))
    }

    /// Returns an iterator over mutable references to the polynomials contained in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for mut polynomial in list.polynomial_iter_mut() {
    ///     polynomial
    ///         .get_mut_monomial(MonomialDegree(0))
    ///         .set_coefficient(10u8);
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(
    ///         *polynomial.get_monomial(MonomialDegree(0)).get_coefficient(),
    ///         10u8
    ///     );
    /// }
    /// assert_eq!(list.polynomial_iter_mut().count(), 4);
    /// ```
    pub fn polynomial_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = Polynomial<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(|sub| Polynomial::from_container(sub.into_container()))
    }

    /// Multiplies (mod $(X^N+1)$), all the polynomials of the list with a unit monomial of a
    /// given degree.
    ///
    /// # Examples
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let mut list = PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6], PolynomialSize(3));
    /// list.update_with_wrapping_monic_monomial_mul(MonomialDegree(2));
    /// let poly = list.get_polynomial(0);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(0)).get_coefficient(), 254);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(1)).get_coefficient(), 253);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(2)).get_coefficient(), 1);
    /// ```
    pub fn update_with_wrapping_monic_monomial_mul<Coef>(&mut self, monomial_degree: MonomialDegree)
    where
        Self: AsMutTensor<Element = Coef>,
        Coef: UnsignedInteger,
    {
        for mut poly in self.polynomial_iter_mut() {
            poly.update_with_wrapping_monic_monomial_mul(monomial_degree);
        }
    }

    /// Divides (mod $(X^N+1)$), all the polynomials of the list with a unit monomial of a
    /// given degree.
    ///
    /// # Examples
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::polynomial::{
    ///     MonomialDegree, PolynomialList,
    /// };
    /// let mut list = PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6], PolynomialSize(3));
    /// list.update_with_wrapping_monic_monomial_div(MonomialDegree(2));
    /// let poly = list.get_polynomial(0);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(0)).get_coefficient(), 3);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(1)).get_coefficient(), 255);
    /// assert_eq!(*poly.get_monomial(MonomialDegree(2)).get_coefficient(), 254);
    /// ```
    pub fn update_with_wrapping_monic_monomial_div<Coef>(&mut self, monomial_degree: MonomialDegree)
    where
        Self: AsMutTensor<Element = Coef>,
        Coef: UnsignedInteger,
    {
        for mut poly in self.polynomial_iter_mut() {
            poly.update_with_wrapping_unit_monomial_div(monomial_degree);
        }
    }
}
