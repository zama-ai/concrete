use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::math::tensor::{
    ck_dim_eq, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::utils::{zip, zip_args};

use super::Complex64;
use concrete_commons::parameters::PolynomialSize;

/// A polynomial in the fourier domain.
///
/// This structure represents a polynomial, which was put in the fourier domain.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct FourierPolynomial<Cont> {
    tensor: Tensor<Cont>,
}

tensor_traits!(FourierPolynomial);

impl FourierPolynomial<AlignedVec<Complex64>> {
    /// Allocates a new empty fourier polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// let fourier_poly = FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(128));
    /// assert_eq!(fourier_poly.polynomial_size(), PolynomialSize(128));
    /// ```
    pub fn allocate(value: Complex64, coef_count: PolynomialSize) -> Self {
        let mut tensor = Tensor::from_container(AlignedVec::new(coef_count.0));
        tensor.fill_with_element(value);
        FourierPolynomial { tensor }
    }
}

impl<Cont> FourierPolynomial<Cont> {
    /// Creates a complex polynomial from an existing container of values.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{
    ///     AlignedVec, Complex64, FourierPolynomial,
    /// };
    /// let mut alvec: AlignedVec<Complex64> = AlignedVec::new(128);
    /// let fourier_poly = FourierPolynomial::from_container(alvec.as_slice_mut());
    /// assert_eq!(fourier_poly.polynomial_size(), PolynomialSize(128));
    /// ```
    pub fn from_container(cont: Cont) -> Self {
        FourierPolynomial {
            tensor: Tensor::from_container(cont),
        }
    }

    pub(crate) fn from_tensor(tensor: Tensor<Cont>) -> Self {
        FourierPolynomial { tensor }
    }

    /// Returns the number of coefficients in the polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// let fourier_poly = FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(128));
    /// assert_eq!(fourier_poly.polynomial_size(), PolynomialSize(128));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize
    where
        Self: AsRefTensor,
    {
        PolynomialSize(self.as_tensor().len())
    }

    /// Returns an iterator over borrowed polynomial coefficients.
    ///
    /// # Note
    ///
    /// We do not give any guarantee on the order of the coefficients.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// let fourier_poly = FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(128));
    /// for coef in fourier_poly.coefficient_iter() {
    ///     assert_eq!(*coef, Complex64::new(0., 0.));
    /// }
    /// assert_eq!(fourier_poly.coefficient_iter().count(), 128);
    /// ```
    pub fn coefficient_iter(&self) -> impl Iterator<Item = &Complex64>
    where
        Self: AsRefTensor<Element = Complex64>,
    {
        self.as_tensor().iter()
    }

    /// Returns an iterator over mutably borrowed polynomial coefficients.
    ///
    /// # Note
    ///
    /// We do not give any guarantee on the order of the coefficients.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// use concrete_core::backends::core::private::math::tensor::AsRefTensor;
    /// let mut fourier_poly = FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(128));
    /// for mut coef in fourier_poly.coefficient_iter_mut() {
    ///     *coef = Complex64::new(1., 1.);
    /// }
    /// assert!(fourier_poly
    ///     .as_tensor()
    ///     .iter()
    ///     .all(|a| *a == Complex64::new(1., 1.)));
    /// assert_eq!(fourier_poly.coefficient_iter_mut().count(), 128);
    /// ```
    pub fn coefficient_iter_mut(&mut self) -> impl Iterator<Item = &mut Complex64>
    where
        Self: AsMutTensor<Element = Complex64>,
    {
        self.as_mut_tensor().iter_mut()
    }

    /// Adds the result of the element-wise product of two polynomials to $(self.len()/2)+2$
    /// elements of the current polynomial:
    /// $$
    /// self\[i\] = self\[i\] + poly_1\[i\] * poly_2\[i\]
    /// $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// let mut fpoly1 = FourierPolynomial::allocate(Complex64::new(1., 2.), PolynomialSize(128));
    /// let fpoly2 = FourierPolynomial::allocate(Complex64::new(3., 4.), PolynomialSize(128));
    /// let fpoly3 = FourierPolynomial::allocate(Complex64::new(5., 6.), PolynomialSize(128));
    /// fpoly1.update_with_multiply_accumulate(&fpoly2, &fpoly3);
    /// // It actually update half+2 elements.
    /// let half = fpoly1.polynomial_size().0 / 2 + 2;
    /// assert!(fpoly1
    ///     .coefficient_iter()
    ///     .take(half)
    ///     .all(|a| *a == Complex64::new(-8., 40.)));
    /// assert!(fpoly1
    ///     .coefficient_iter()
    ///     .skip(half)
    ///     .all(|a| *a == Complex64::new(1., 2.)));
    /// ```
    pub fn update_with_multiply_accumulate<PolyCont1, PolyCont2>(
        &mut self,
        poly_1: &FourierPolynomial<PolyCont1>,
        poly_2: &FourierPolynomial<PolyCont2>,
    ) where
        Self: AsMutTensor<Element = Complex64>,
        FourierPolynomial<PolyCont1>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<PolyCont2>: AsRefTensor<Element = Complex64>,
    {
        ck_dim_eq!(self.polynomial_size().0 => poly_1.polynomial_size().0, poly_2.polynomial_size().0);
        #[cfg(not(target_feature = "avx2"))]
        regular_uhwap(self, poly_1, poly_2);
        #[cfg(target_feature = "avx2")]
        avx2_uhwap(self, poly_1, poly_2);
    }
    /// Adds the result of the element-wise product of `poly_1` with `poly_2`, and the result
    /// of the element-wise product of `poly_3` with `poly_4`, to $(self.len()/2)+2$ elements of
    /// the current polynomial:
    /// $$
    /// self\[i\] = self\[i\] + poly_1\[i\] * poly_2\[i\] + poly_3\[i\] * poly_4\[i\]
    /// $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// let mut fpoly1 = FourierPolynomial::allocate(Complex64::new(1., 2.), PolynomialSize(128));
    /// let fpoly2 = FourierPolynomial::allocate(Complex64::new(3., 4.), PolynomialSize(128));
    /// let fpoly3 = FourierPolynomial::allocate(Complex64::new(5., 6.), PolynomialSize(128));
    /// let fpoly4 = FourierPolynomial::allocate(Complex64::new(7., 8.), PolynomialSize(128));
    /// let fpoly5 = FourierPolynomial::allocate(Complex64::new(9., 10.), PolynomialSize(128));
    /// fpoly1.update_with_two_multiply_accumulate(&fpoly2, &fpoly3, &fpoly4, &fpoly5);
    /// // It actually update half+2 elements.
    /// let half = fpoly1.polynomial_size().0 / 2 + 2;
    /// assert!(fpoly1
    ///     .coefficient_iter()
    ///     .take(half)
    ///     .all(|a| *a == Complex64::new(-25., 182.)));
    /// assert!(fpoly1
    ///     .coefficient_iter()
    ///     .skip(half)
    ///     .all(|a| *a == Complex64::new(1., 2.)));
    /// ```
    pub fn update_with_two_multiply_accumulate<Cont1, Cont2, Cont3, Cont4>(
        &mut self,
        poly_1: &FourierPolynomial<Cont1>,
        poly_2: &FourierPolynomial<Cont2>,
        poly_3: &FourierPolynomial<Cont3>,
        poly_4: &FourierPolynomial<Cont4>,
    ) where
        Self: AsMutTensor<Element = Complex64>,
        FourierPolynomial<Cont1>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<Cont2>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<Cont3>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<Cont4>: AsRefTensor<Element = Complex64>,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            poly_1.polynomial_size().0,
            poly_2.polynomial_size().0,
            poly_3.polynomial_size().0,
            poly_4.polynomial_size().0
        );
        #[cfg(not(target_feature = "avx2"))]
        regular_uhwatp(self, poly_1, poly_2, poly_3, poly_4);
        #[cfg(target_feature = "avx2")]
        avx2_uhwatp(self, poly_1, poly_2, poly_3, poly_4);
    }

    /// Updates two polynomials with the following operation:
    ///
    /// $$
    /// result_1\[i\]=result_1\[i\]+poly_{a_1}\[i\]*poly_b\[i\]+poly_{c_1}\[i\]*poly_d\[i\]\\\\
    /// result_2\[i\]=result_2\[i\]+poly_{a_2}\[i\]*poly_b\[i\]+poly_{c_2}\[i\]*poly_d\[i\]
    /// $$
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial};
    /// macro_rules! new_poly {
    ///     ($name: ident, $re: literal, $im: literal) => {
    ///         let mut $name =
    ///             FourierPolynomial::allocate(Complex64::new($re, $im), PolynomialSize(128));
    ///     };
    /// }
    /// new_poly!(fpoly_1, 1., 2.);
    /// new_poly!(fpoly_2, 3., 4.);
    /// new_poly!(fpoly_3, 5., 6.);
    /// new_poly!(fpoly_4, 7., 8.);
    /// new_poly!(fpoly_5, 9., 10.);
    /// new_poly!(fpoly_6, 11., 12.);
    /// new_poly!(fpoly_7, 13., 14.);
    /// new_poly!(fpoly_8, 15., 16.);
    /// FourierPolynomial::update_two_with_two_multiply_accumulate(
    ///     &mut fpoly_1,
    ///     &mut fpoly_2,
    ///     &fpoly_3,
    ///     &fpoly_4,
    ///     &fpoly_5,
    ///     &fpoly_6,
    ///     &fpoly_7,
    ///     &fpoly_8,
    /// );
    /// // It actually update half+2 elements.
    /// let half = fpoly_1.polynomial_size().0 / 2 + 2;
    /// assert!(fpoly_1
    ///     .coefficient_iter()
    ///     .take(half)
    ///     .all(|a| *a == Complex64::new(-41., 462.)));
    /// assert!(fpoly_1
    ///     .coefficient_iter()
    ///     .skip(half)
    ///     .all(|a| *a == Complex64::new(1., 2.)));
    /// assert!(fpoly_2
    ///     .coefficient_iter()
    ///     .take(half)
    ///     .all(|a| *a == Complex64::new(-43., 564.)));
    /// assert!(fpoly_2
    ///     .coefficient_iter()
    ///     .skip(half)
    ///     .all(|a| *a == Complex64::new(3., 4.)));
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn update_two_with_two_multiply_accumulate<C2, C3, C4, C5, C6, C7, C8>(
        result_1: &mut FourierPolynomial<Cont>,
        result_2: &mut FourierPolynomial<C2>,
        poly_a_1: &FourierPolynomial<C3>,
        poly_a_2: &FourierPolynomial<C7>,
        poly_b: &FourierPolynomial<C4>,
        poly_c_1: &FourierPolynomial<C5>,
        poly_c_2: &FourierPolynomial<C8>,
        poly_d: &FourierPolynomial<C6>,
    ) where
        FourierPolynomial<Cont>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<C2>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<C4>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<C5>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<C6>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<C7>: AsRefTensor<Element = Complex64>,
        FourierPolynomial<C8>: AsRefTensor<Element = Complex64>,
    {
        ck_dim_eq!(result_1.polynomial_size().0 =>
            result_2.polynomial_size().0,
            poly_a_1.polynomial_size().0,
            poly_b.polynomial_size().0,
            poly_c_1.polynomial_size().0,
            poly_d.polynomial_size().0,
            poly_a_2.polynomial_size().0,
            poly_c_2.polynomial_size().0
        );
        #[cfg(not(target_feature = "avx2"))]
        regular_uthwatp(
            result_1, result_2, poly_a_1, poly_a_2, poly_b, poly_c_1, poly_c_2, poly_d,
        );
        #[cfg(target_feature = "avx2")]
        avx2_uthwatp(
            result_1, result_2, poly_a_1, poly_a_2, poly_b, poly_c_1, poly_c_2, poly_d,
        );
    }
}

#[allow(unused)]
fn regular_uhwap<C1, C2, C3>(
    res: &mut FourierPolynomial<C1>,
    poly_1: &FourierPolynomial<C2>,
    poly_2: &FourierPolynomial<C3>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
{
    let half = res.polynomial_size().0 / 2 + 2;
    for zip_args!(res, coef_1, coef_2) in zip!(
        res.as_mut_tensor().iter_mut().take(half),
        poly_1.as_tensor().iter().take(half),
        poly_2.as_tensor().iter().take(half)
    ) {
        *res += coef_1 * coef_2;
    }
}

#[allow(unused)]
fn avx2_uhwap<C1, C2, C3>(
    res: &mut FourierPolynomial<C1>,
    poly_1: &FourierPolynomial<C2>,
    poly_2: &FourierPolynomial<C3>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[allow(unused_imports)]
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = poly_1.as_tensor().len();
    let index = n / 2 + 2;
    for zip_args!(coeff_tmp_ref, coeff_a_i, coeff_b_i) in zip!(
        res.as_mut_tensor().as_mut_slice()[..index].chunks_mut(2),
        poly_1.as_tensor().as_slice()[..index].chunks(2),
        poly_2.as_tensor().as_slice()[..index].chunks(2)
    ) {
        unsafe {
            let vec_a: __m256d = _mm256_setr_pd(
                coeff_a_i[0].re,
                coeff_a_i[0].im,
                coeff_a_i[1].re,
                coeff_a_i[1].im,
            );
            let mut vec_b: __m256d = _mm256_setr_pd(
                coeff_b_i[0].re,
                coeff_b_i[0].im,
                coeff_b_i[1].re,
                coeff_b_i[1].im,
            );

            let neg: __m256d = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            let vec_ab = _mm256_mul_pd(vec_a, vec_b);
            vec_b = _mm256_permute_pd(vec_b, 0x5);
            vec_b = _mm256_mul_pd(vec_b, neg);
            let vec_ab_bis = _mm256_mul_pd(vec_a, vec_b);
            let resref: *mut __m256d = coeff_tmp_ref.as_mut_ptr() as *mut __m256d;
            *resref = _mm256_add_pd(_mm256_hsub_pd(vec_ab, vec_ab_bis), *resref);
        }
    }
}

#[allow(unused)]
fn regular_uhwatp<C1, C2, C3, C4, C5>(
    res: &mut FourierPolynomial<C1>,
    poly_1: &FourierPolynomial<C2>,
    poly_2: &FourierPolynomial<C3>,
    poly_3: &FourierPolynomial<C4>,
    poly_4: &FourierPolynomial<C5>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C4>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C5>: AsRefTensor<Element = Complex64>,
{
    let half = res.polynomial_size().0 / 2 + 2;
    for zip_args!(res_ref, a_i, b_i, c_i, d_i) in zip!(
        res.as_mut_tensor().iter_mut().take(half),
        poly_1.as_tensor().iter().take(half),
        poly_2.as_tensor().iter().take(half),
        poly_3.as_tensor().iter().take(half),
        poly_4.as_tensor().iter().take(half)
    ) {
        *res_ref += a_i * b_i + c_i * d_i;
    }
}

#[allow(unused)]
fn avx2_uhwatp<C1, C2, C3, C4, C5>(
    res: &mut FourierPolynomial<C1>,
    poly_1: &FourierPolynomial<C2>,
    poly_2: &FourierPolynomial<C3>,
    poly_3: &FourierPolynomial<C4>,
    poly_4: &FourierPolynomial<C5>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C4>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C5>: AsRefTensor<Element = Complex64>,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[allow(unused_imports)]
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = poly_1.as_tensor().len();
    let index = n / 2 + 2;
    for zip_args!(res_ref, a_i, b_i, c_i, d_i) in zip!(
        res.as_mut_tensor().as_mut_slice()[..index].chunks_mut(2),
        poly_1.as_tensor().as_slice()[..index].chunks(2),
        poly_2.as_tensor().as_slice()[..index].chunks(2),
        poly_3.as_tensor().as_slice()[..index].chunks(2),
        poly_4.as_tensor().as_slice()[..index].chunks(2)
    ) {
        // *res_ref += a_i * b_i + c_i * d_i;
        unsafe {
            let vec_a: __m256d = _mm256_setr_pd(a_i[0].re, a_i[0].im, a_i[1].re, a_i[1].im);
            let mut vec_b: __m256d = _mm256_setr_pd(b_i[0].re, b_i[0].im, b_i[1].re, b_i[1].im);
            let vec_c: __m256d = _mm256_setr_pd(c_i[0].re, c_i[0].im, c_i[1].re, c_i[1].im);
            let mut vec_d: __m256d = _mm256_setr_pd(d_i[0].re, d_i[0].im, d_i[1].re, d_i[1].im);

            let neg: __m256d = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
            let vec_ab = _mm256_mul_pd(vec_a, vec_b);
            let vec_cd = _mm256_mul_pd(vec_c, vec_d);

            vec_b = _mm256_permute_pd(vec_b, 0x5);
            vec_b = _mm256_mul_pd(vec_b, neg);
            vec_d = _mm256_permute_pd(vec_d, 0x5);
            vec_d = _mm256_mul_pd(vec_d, neg);

            let vec_ab_bis = _mm256_mul_pd(vec_a, vec_b);
            let vec_cd_bis = _mm256_mul_pd(vec_c, vec_d);

            let res_ref_avx: *mut __m256d = res_ref.as_mut_ptr() as *mut __m256d;
            *res_ref_avx = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_hsub_pd(vec_ab, vec_ab_bis),
                    _mm256_hsub_pd(vec_cd, vec_cd_bis),
                ),
                *res_ref_avx,
            );
        }
    }
}

#[allow(unused, clippy::too_many_arguments)]
fn regular_uthwatp<C1, C2, C3, C4, C5, C6, C7, C8>(
    result_1: &mut FourierPolynomial<C1>,
    result_2: &mut FourierPolynomial<C2>,
    poly_a_1: &FourierPolynomial<C3>,
    poly_a_2: &FourierPolynomial<C4>,
    poly_b: &FourierPolynomial<C5>,
    poly_c_1: &FourierPolynomial<C6>,
    poly_c_2: &FourierPolynomial<C7>,
    poly_d: &FourierPolynomial<C8>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C4>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C5>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C6>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C7>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C8>: AsRefTensor<Element = Complex64>,
{
    let n = poly_a_1.as_tensor().len();
    let index = n / 2 + 2;
    for zip_args!(res_1_i, a_1_i, b_i, c_1_i, d_i, res_2_i, a_2_i, c_2_i) in zip!(
        result_1.as_mut_tensor().as_mut_slice()[..index].iter_mut(),
        poly_a_1.as_tensor().as_slice()[..index].iter(),
        poly_b.as_tensor().as_slice()[..index].iter(),
        poly_c_1.as_tensor().as_slice()[..index].iter(),
        poly_d.as_tensor().as_slice()[..index].iter(),
        result_2.as_mut_tensor().as_mut_slice()[..index].iter_mut(),
        poly_a_2.as_tensor().as_slice()[..index].iter(),
        poly_c_2.as_tensor().as_slice()[..index].iter()
    ) {
        *res_1_i += a_1_i * b_i + c_1_i * d_i;
        *res_2_i += a_2_i * b_i + c_2_i * d_i;
    }
}

#[allow(unused, clippy::too_many_arguments)]
fn avx2_uthwatp<C1, C2, C3, C4, C5, C6, C7, C8>(
    result_1: &mut FourierPolynomial<C1>,
    result_2: &mut FourierPolynomial<C2>,
    poly_a_1: &FourierPolynomial<C3>,
    poly_a_2: &FourierPolynomial<C4>,
    poly_b: &FourierPolynomial<C5>,
    poly_c_1: &FourierPolynomial<C6>,
    poly_c_2: &FourierPolynomial<C7>,
    poly_d: &FourierPolynomial<C8>,
) where
    FourierPolynomial<C1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C2>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<C3>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C4>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C5>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C6>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C7>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<C8>: AsRefTensor<Element = Complex64>,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[allow(unused_imports)]
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = result_1.as_tensor().len();
    let index = n / 2 + 2;
    // we take complex two by two
    for zip_args!(res_1_i, a_1_i, b_i, c_1_i, d_i, res_2_i, a_2_i, c_2_i) in zip!(
        result_1.as_mut_tensor().as_mut_slice()[..index].chunks_mut(2),
        poly_a_1.as_tensor().as_slice()[..index].chunks(2),
        poly_b.as_tensor().as_slice()[..index].chunks(2),
        poly_c_1.as_tensor().as_slice()[..index].chunks(2),
        poly_d.as_tensor().as_slice()[..index].chunks(2),
        result_2.as_mut_tensor().as_mut_slice()[..index].chunks_mut(2),
        poly_a_2.as_tensor().as_slice()[..index].chunks(2),
        poly_c_2.as_tensor().as_slice()[..index].chunks(2)
    ) {
        // create avx vectors
        unsafe {
            let vec_a1: __m256d = _mm256_setr_pd(
                a_1_i.get_unchecked(0).re,
                a_1_i.get_unchecked(0).im,
                a_1_i.get_unchecked(1).re,
                a_1_i.get_unchecked(1).im,
            );
            let vec_a2: __m256d = _mm256_setr_pd(
                a_2_i.get_unchecked(0).re,
                a_2_i.get_unchecked(0).im,
                a_2_i.get_unchecked(1).re,
                a_2_i.get_unchecked(1).im,
            );
            let vec_c1: __m256d = _mm256_setr_pd(
                c_1_i.get_unchecked(0).re,
                c_1_i.get_unchecked(0).im,
                c_1_i.get_unchecked(1).re,
                c_1_i.get_unchecked(1).im,
            );
            let vec_c2: __m256d = _mm256_setr_pd(
                c_2_i.get_unchecked(0).re,
                c_2_i.get_unchecked(0).im,
                c_2_i.get_unchecked(1).re,
                c_2_i.get_unchecked(1).im,
            );
            let mut vec_b: __m256d = _mm256_setr_pd(
                b_i.get_unchecked(0).re,
                b_i.get_unchecked(0).im,
                b_i.get_unchecked(1).re,
                b_i.get_unchecked(1).im,
            );
            let mut vec_d: __m256d = _mm256_setr_pd(
                d_i.get_unchecked(0).re,
                d_i.get_unchecked(0).im,
                d_i.get_unchecked(1).re,
                d_i.get_unchecked(1).im,
            );
            let neg: __m256d = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

            // Multiply
            let vec_a1b: __m256d = _mm256_mul_pd(vec_a1, vec_b);
            let vec_a2b: __m256d = _mm256_mul_pd(vec_a2, vec_b);
            let vec_c1d: __m256d = _mm256_mul_pd(vec_c1, vec_d);
            let vec_c2d: __m256d = _mm256_mul_pd(vec_c2, vec_d);

            // Permute the real and imaginary elements of glwe
            vec_b = _mm256_permute_pd(vec_b, 0x5);
            vec_d = _mm256_permute_pd(vec_d, 0x5);

            // Negate the imaginary elements of glwe
            vec_b = _mm256_mul_pd(vec_b, neg);
            vec_d = _mm256_mul_pd(vec_d, neg);

            // Multiply
            let vec_a1b_bis: __m256d = _mm256_mul_pd(vec_a1, vec_b);
            let vec_a2b_bis: __m256d = _mm256_mul_pd(vec_a2, vec_b);
            let vec_c1d_bis: __m256d = _mm256_mul_pd(vec_c1, vec_d);
            let vec_c2d_bis: __m256d = _mm256_mul_pd(vec_c2, vec_d);

            let res_1_ref = res_1_i.as_mut_ptr() as *mut __m256d;
            *res_1_ref = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_hsub_pd(vec_a1b, vec_a1b_bis),
                    _mm256_hsub_pd(vec_c1d, vec_c1d_bis),
                ),
                *res_1_ref,
            );
            let res_2_ref = res_2_i.as_mut_ptr() as *mut __m256d;

            *res_2_ref = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_hsub_pd(vec_a2b, vec_a2b_bis),
                    _mm256_hsub_pd(vec_c2d, vec_c2d_bis),
                ),
                *res_2_ref,
            );
        }
    }
}
