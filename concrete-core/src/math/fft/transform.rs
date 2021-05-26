use std::slice;

use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::{c64, Flag, Sign};

use crate::crypto::UnsignedTorus;
use crate::math::fft::twiddles::{BackwardCorrector, ForwardCorrector};
use crate::math::polynomial::{Polynomial, PolynomialSize};
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefTensor};
use crate::numeric::{CastInto, SignedInteger, UnsignedInteger};
use crate::{ck_dim_eq, zip};

use super::{Complex64, Correctors, FourierPolynomial};

/// A fast fourier transformer.
///
/// This transformer type allows to send polynomials of a fixed size, back and forth in the fourier
/// domain.
pub struct Fft {
    forward_plan: C2CPlan64,
    backward_plan: C2CPlan64,
    correctors: Correctors,
    temporary: FourierPolynomial<AlignedVec<Complex64>>,
}

impl Fft {
    /// Generates a new transformer for polynomials a given size.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::Fft;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let fft = Fft::new(PolynomialSize(256));
    /// assert_eq!(fft.polynomial_size(), PolynomialSize(256));
    /// ```
    pub fn new(size: PolynomialSize) -> Fft {
        debug_assert!(
            [256, 512, 1024, 2048, 4096].contains(&size.0),
            "The size chosen is not valid ({}). Should be 256, 512, 1024, 2048 or 4096",
            size.0
        );
        let forward_plan = C2CPlan64::aligned(&[size.0], Sign::Forward, Flag::MEASURE).unwrap();
        let backward_plan = C2CPlan64::aligned(&[size.0], Sign::Backward, Flag::MEASURE).unwrap();
        let temporary = FourierPolynomial::allocate(Complex64::new(0., 0.), PolynomialSize(size.0));
        let correctors = Correctors::new(size.0);
        Fft {
            forward_plan,
            backward_plan,
            correctors,
            temporary,
        }
    }

    /// Returns the polynomial size accepted by this transformer.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::Fft;
    /// use concrete_core::math::polynomial::PolynomialSize;
    /// let fft = Fft::new(PolynomialSize(256));
    /// assert_eq!(fft.polynomial_size(), PolynomialSize(256));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.temporary.polynomial_size()
    }

    /// Performs the forward fourier transform of the `poly` polynomial, viewed as a polynomial of
    /// torus coefficients, and stores the result in `fourier_poly`.
    ///
    /// # Note
    ///
    /// It should be noted that this method is subotpimal, as it only uses half of the computational
    /// power of the transformer. For a faster approach, you should consider processing the
    /// polynomials two by two with the [`Fft::forward_two_as_torus`] method.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::{Fft, FourierPolynomial, Complex64};
    /// use concrete_core::math::polynomial::{Polynomial, PolynomialSize};
    /// use concrete_core::math::random::RandomGenerator;
    /// use concrete_core::math::tensor::AsRefTensor;
    /// use concrete_core::crypto::UnsignedTorus;
    /// let mut generator = RandomGenerator::new(None);
    /// let mut fft = Fft::new(PolynomialSize(256));
    /// let mut fourier_poly = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut poly = Polynomial::allocate(0u32, PolynomialSize(256));
    /// generator.fill_tensor_with_random_uniform(&mut poly);
    /// fft.forward_as_torus(&mut fourier_poly, &poly);
    /// let mut out = Polynomial::allocate(0u32, PolynomialSize(256));
    /// fft.add_backward_as_torus(&mut out, &mut fourier_poly);
    /// out.as_tensor()
    ///    .iter()
    ///    .zip(poly.as_tensor().iter())
    ///    .for_each(|(output, expected)| assert_eq!(*output, *expected));
    /// ```
    pub fn forward_as_torus<OutCont, InCont, Coef>(
        &mut self,
        fourier_poly: &mut FourierPolynomial<OutCont>,
        poly: &Polynomial<InCont>,
    ) where
        FourierPolynomial<OutCont>: AsMutTensor<Element = Complex64>,
        Polynomial<InCont>: AsRefTensor<Element = Coef>,
        Coef: UnsignedTorus,
    {
        self.forward(fourier_poly, poly, regular_convert_forward_single_torus);
    }

    /// Performs the forward fourier transform of the `poly_1` and `poly_2` polynomials, viewed
    /// as polynomials of torus coefficients, and stores the result in `fourier_poly_1` and
    /// `fourier_poly_2`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::{Fft, FourierPolynomial, Complex64};
    /// use concrete_core::math::polynomial::{Polynomial, PolynomialSize};
    /// use concrete_core::math::random::RandomGenerator;
    /// use concrete_core::math::tensor::AsRefTensor;
    /// use concrete_core::crypto::UnsignedTorus;
    /// use concrete_core::numeric::UnsignedInteger;
    /// let mut generator = RandomGenerator::new(None);
    /// let mut fft = Fft::new(PolynomialSize(256));
    /// let mut fourier_poly_1 = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut fourier_poly_2 = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut poly_1 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// let mut poly_2 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// generator.fill_tensor_with_random_uniform(&mut poly_1);
    /// generator.fill_tensor_with_random_uniform(&mut poly_2);
    /// fft.forward_two_as_torus(&mut fourier_poly_1, &mut fourier_poly_2, &poly_1, &poly_2);
    /// let mut out_1 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// let mut out_2 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// fft.add_backward_two_as_torus(
    ///     &mut out_1,
    ///     &mut out_2,
    ///     &mut fourier_poly_1,
    ///     &mut fourier_poly_2
    /// );
    /// out_1.as_tensor()
    ///    .iter()
    ///    .zip(poly_1.as_tensor().iter())
    ///    .for_each(|(out, exp)| assert_eq!(out, exp));
    /// out_2.as_tensor()
    ///    .iter()
    ///    .zip(poly_2.as_tensor().iter())
    ///    .for_each(|(out, exp)| assert_eq!(out, exp));
    /// ```
    pub fn forward_two_as_torus<InCont1, InCont2, OutCont1, OutCont2, Coef>(
        &mut self,
        fourier_poly_1: &mut FourierPolynomial<OutCont1>,
        fourier_poly_2: &mut FourierPolynomial<OutCont2>,
        poly_1: &Polynomial<InCont1>,
        poly_2: &Polynomial<InCont2>,
    ) where
        Polynomial<InCont1>: AsRefTensor<Element = Coef>,
        Polynomial<InCont2>: AsRefTensor<Element = Coef>,
        FourierPolynomial<OutCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<OutCont2>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedTorus,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );
        self.forward_two(
            fourier_poly_1,
            fourier_poly_2,
            poly_1,
            poly_2,
            regular_convert_forward_two_torus,
        );
    }

    /// Performs the forward fourier transform of the `poly` polynomial, viewed as a polynomial of
    /// integer coefficients, and stores the result in `fourier_poly`.
    ///
    /// # Note
    ///
    /// It should be noted that this method is subotpimal, as it only uses half of the computational
    /// power of the transformer. For a faster approach, you should consider processing the
    /// polynomials two by two with the [`Fft::forward_two_as_integer`] method.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::{Fft, FourierPolynomial, Complex64};
    /// use concrete_core::math::polynomial::{Polynomial, PolynomialSize};
    /// use concrete_core::math::random::RandomGenerator;
    /// use concrete_core::math::tensor::AsRefTensor;
    /// use concrete_core::numeric::UnsignedInteger;
    /// let mut generator = RandomGenerator::new(None);
    /// let mut fft = Fft::new(PolynomialSize(256));
    /// let mut fourier_poly = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut poly = Polynomial::allocate(0u32, PolynomialSize(256));
    /// generator.fill_tensor_with_random_uniform(&mut poly);
    /// fft.forward_as_integer(&mut fourier_poly, &poly);
    /// let mut out = Polynomial::allocate(0u32, PolynomialSize(256));
    /// fft.add_backward_as_integer(&mut out, &mut fourier_poly);
    /// out.as_tensor()
    ///    .iter()
    ///    .zip(poly.as_tensor().iter())
    ///    .for_each(|(out, exp)| assert_eq!(*out, *exp));
    /// ```
    pub fn forward_as_integer<OutCont, InCont, Coef>(
        &mut self,
        fourier_poly: &mut FourierPolynomial<OutCont>,
        poly: &Polynomial<InCont>,
    ) where
        FourierPolynomial<OutCont>: AsMutTensor<Element = Complex64>,
        Polynomial<InCont>: AsRefTensor<Element = Coef>,
        Coef: UnsignedInteger,
    {
        self.forward(fourier_poly, poly, regular_convert_forward_single_integer);
    }

    /// Performs the forward fourier transform of the `poly_1` and `poly_2` polynomials, viewed
    /// as polynomials of integer coefficients, and stores the result in `fourier_poly_1` and
    /// `fourier_poly_2`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_core::math::fft::{Fft, FourierPolynomial, Complex64};
    /// use concrete_core::math::polynomial::{Polynomial, PolynomialSize};
    /// use concrete_core::math::random::RandomGenerator;
    /// use concrete_core::math::tensor::AsRefTensor;
    /// use concrete_core::numeric::UnsignedInteger;
    /// let mut generator = RandomGenerator::new(None);
    /// let mut fft = Fft::new(PolynomialSize(256));
    /// let mut fourier_poly_1 = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut fourier_poly_2 = FourierPolynomial::allocate(
    ///     Complex64::new(0.,0.),
    ///     PolynomialSize(256)
    /// );
    /// let mut poly_1 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// let mut poly_2 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// generator.fill_tensor_with_random_uniform(&mut poly_1);
    /// generator.fill_tensor_with_random_uniform(&mut poly_2);
    /// fft.forward_two_as_integer(&mut fourier_poly_1, &mut fourier_poly_2, &poly_1, &poly_2);
    /// let mut out_1 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// let mut out_2 = Polynomial::allocate(0u32, PolynomialSize(256));
    /// fft.add_backward_two_as_integer(
    ///     &mut out_1,
    ///     &mut out_2,
    ///     &mut fourier_poly_1,
    ///     &mut fourier_poly_2
    /// );
    /// out_1.as_tensor()
    ///    .iter()
    ///    .zip(poly_1.as_tensor().iter())
    ///    .for_each(|(out, exp)| assert_eq!(out, exp));
    /// out_2.as_tensor()
    ///    .iter()
    ///    .zip(poly_2.as_tensor().iter())
    ///    .for_each(|(out, exp)| assert_eq!(out, exp));
    /// ```
    pub fn forward_two_as_integer<InCont1, InCont2, OutCont1, OutCont2, Coef>(
        &mut self,
        fourier_poly_1: &mut FourierPolynomial<OutCont1>,
        fourier_poly_2: &mut FourierPolynomial<OutCont2>,
        poly_1: &Polynomial<InCont1>,
        poly_2: &Polynomial<InCont2>,
    ) where
        Polynomial<InCont1>: AsRefTensor<Element = Coef>,
        Polynomial<InCont2>: AsRefTensor<Element = Coef>,
        FourierPolynomial<OutCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<OutCont2>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedInteger,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );
        self.forward_two(
            fourier_poly_1,
            fourier_poly_2,
            poly_1,
            poly_2,
            regular_convert_forward_two_integer,
        );
    }

    /// Performs the backward fourier transform of the `fourier_poly` polynomial, viewed as a
    /// polynomial of torus coefficients, and adds the result to `poly`.
    ///
    /// See [`Fft::forward_as_torus`] for an example.
    ///
    /// # Note
    ///
    /// It should be noted that this method is subotpimal, as it only uses half of the computational
    /// power of the transformer. For a faster approach, you should consider processing the
    /// polynomials two by two with the [`Fft::add_backward_two_as_torus`] method.
    pub fn add_backward_as_torus<OutCont, InCont, Coef>(
        &mut self,
        poly: &mut Polynomial<OutCont>,
        fourier_poly: &mut FourierPolynomial<InCont>,
    ) where
        Polynomial<OutCont>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedTorus,
    {
        ck_dim_eq!(self.polynomial_size().0 => fourier_poly.polynomial_size().0, poly.polynomial_size().0);
        self.backward(
            poly,
            fourier_poly,
            regular_convert_add_backward_single_torus,
        );
    }

    /// Performs the backward fourier transform of the `fourier_poly` polynomial, viewed as a
    /// polynomial of integer coefficients, and adds the result to `poly`.
    ///
    /// See [`Fft::forward_as_integer`] for an example.
    ///
    /// # Note
    ///
    /// It should be noted that this method is subotpimal, as it only uses half of the computational
    /// power of the transformer. For a faster approach, you should consider processing the
    /// polynomials two by two with the [`Fft::add_backward_two_as_integer`] method.
    pub fn add_backward_as_integer<OutCont, InCont, Coef>(
        &mut self,
        poly: &mut Polynomial<OutCont>,
        fourier_poly: &mut FourierPolynomial<InCont>,
    ) where
        Polynomial<OutCont>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedInteger,
    {
        ck_dim_eq!(self.polynomial_size().0 => fourier_poly.polynomial_size().0, poly.polynomial_size().0);
        self.backward(
            poly,
            fourier_poly,
            regular_convert_add_backward_single_integer,
        );
    }

    /// Performs the backward fourier transform of the `fourier_poly_1` and `fourier_poly_2`
    /// polynomials, viewed as polynomials of torus elements, and adds the result to the  
    /// `poly_1` and `poly_2` polynomials.
    ///
    /// See [`Fft::forward_two_as_torus`] for an example.
    pub fn add_backward_two_as_torus<OutCont1, OutCont2, InCont1, InCont2, Coef>(
        &mut self,
        poly_1: &mut Polynomial<OutCont1>,
        poly_2: &mut Polynomial<OutCont2>,
        fourier_poly_1: &mut FourierPolynomial<InCont1>,
        fourier_poly_2: &mut FourierPolynomial<InCont2>,
    ) where
        Polynomial<OutCont1>: AsMutTensor<Element = Coef>,
        Polynomial<OutCont2>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<InCont2>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedTorus,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );
        self.backward_two(
            poly_1,
            poly_2,
            fourier_poly_1,
            fourier_poly_2,
            regular_convert_add_backward_two_torus,
        );
    }

    /// Performs the backward fourier transform of the `fourier_poly_1` and `fourier_poly_2`
    /// polynomials, viewed as polynomials of integer coefficients, and adds the result to the  
    /// `poly_1` and `poly_2` polynomials.
    ///
    /// See [`Fft::forward_two_as_integer`] for an example.
    pub fn add_backward_two_as_integer<OutCont1, OutCont2, InCont1, InCont2, Coef>(
        &mut self,
        poly_1: &mut Polynomial<OutCont1>,
        poly_2: &mut Polynomial<OutCont2>,
        fourier_poly_1: &mut FourierPolynomial<InCont1>,
        fourier_poly_2: &mut FourierPolynomial<InCont2>,
    ) where
        Polynomial<OutCont1>: AsMutTensor<Element = Coef>,
        Polynomial<OutCont2>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<InCont2>: AsMutTensor<Element = Complex64>,
        Coef: UnsignedInteger,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );
        self.backward_two(
            poly_1,
            poly_2,
            fourier_poly_1,
            fourier_poly_2,
            regular_convert_add_backward_two_integer,
        );
    }

    pub(super) fn forward<OutCont, InCont, Coef>(
        &mut self,
        fourier_poly: &mut FourierPolynomial<OutCont>,
        poly: &Polynomial<InCont>,
        convert_function: impl Fn(
            &mut FourierPolynomial<AlignedVec<Complex64>>,
            &Polynomial<InCont>,
            &ForwardCorrector<&'static [Complex64]>,
        ),
    ) where
        Polynomial<InCont>: AsRefTensor<Element = Coef>,
        FourierPolynomial<OutCont>: AsMutTensor<Element = Complex64>,
    {
        ck_dim_eq!(self.polynomial_size().0 => fourier_poly.polynomial_size().0, poly.polynomial_size().0);

        // We convert the data to real and fill the temporary buffer
        convert_function(&mut self.temporary, &poly, &self.correctors.forward);

        // We perform the forward fft
        self.forward_plan
            .c2c(
                &mut self.temporary.as_mut_tensor().as_mut_slice(),
                &mut fourier_poly.as_mut_tensor().as_mut_slice(),
            )
            .expect("forward: fft.c2c threw an error...");
    }

    pub(super) fn forward_two<InCont1, InCont2, OutCont1, OutCont2, Coef>(
        &mut self,
        fourier_poly_1: &mut FourierPolynomial<OutCont1>,
        fourier_poly_2: &mut FourierPolynomial<OutCont2>,
        poly_1: &Polynomial<InCont1>,
        poly_2: &Polynomial<InCont2>,
        convert_function: impl Fn(
            &mut FourierPolynomial<AlignedVec<Complex64>>,
            &Polynomial<InCont1>,
            &Polynomial<InCont2>,
            &ForwardCorrector<&'static [Complex64]>,
        ),
    ) where
        Polynomial<InCont1>: AsRefTensor<Element = Coef>,
        Polynomial<InCont2>: AsRefTensor<Element = Coef>,
        FourierPolynomial<OutCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<OutCont2>: AsMutTensor<Element = Complex64>,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );

        convert_function(
            &mut self.temporary,
            &poly_1,
            &poly_2,
            &self.correctors.forward,
        );

        // We perform the forward on the first fourier polynomial.
        self.forward_plan
            .c2c(
                &mut self.temporary.as_mut_tensor().as_mut_slice(),
                &mut fourier_poly_1.as_mut_tensor().as_mut_slice(),
            )
            .expect("forward_two: fft.c2c threw an error...");

        // We replicate the coefficients on the second fourier polynomial.
        replicate_coefficients(
            &mut fourier_poly_1.as_mut_tensor().as_mut_slice(),
            &mut fourier_poly_2.as_mut_tensor().as_mut_slice(),
            self.polynomial_size().0,
        );
    }

    pub(super) fn backward<OutCont, InCont, Coef>(
        &mut self,
        poly: &mut Polynomial<OutCont>,
        fourier_poly: &mut FourierPolynomial<InCont>,
        convert_function: impl Fn(
            &mut Polynomial<OutCont>,
            &FourierPolynomial<AlignedVec<Complex64>>,
            &BackwardCorrector<&'static [Complex64]>,
        ),
    ) where
        Polynomial<OutCont>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont>: AsMutTensor<Element = Complex64>,
    {
        // We propagate the values to their conjugates that were not computed.
        let mut first_view = fourier_poly.as_mut_tensor().as_mut_slice();
        let (b_first, b_second) = split_in_imut_mut(&mut first_view, self.polynomial_size().0);
        for (fft_bj, rot_fft_bj) in zip!(b_first.iter(), b_second.iter_mut().rev()) {
            *rot_fft_bj = fft_bj.conj();
        }

        // We perform the backward fft
        self.backward_plan
            .c2c(
                &mut fourier_poly.as_mut_tensor().as_mut_slice(),
                &mut self.temporary.as_mut_tensor().as_mut_slice(),
            )
            .expect("put_in_coeff_domain: fft.c2c threw an error...");

        // We fill the polynomial with the conversion function
        convert_function(poly, &self.temporary, &self.correctors.backward)
    }

    pub(super) fn backward_two<OutCont1, OutCont2, InCont1, InCont2, Coef>(
        &mut self,
        poly_1: &mut Polynomial<OutCont1>,
        poly_2: &mut Polynomial<OutCont2>,
        fourier_poly_1: &mut FourierPolynomial<InCont1>,
        fourier_poly_2: &mut FourierPolynomial<InCont2>,
        convert_function: impl Fn(
            &mut Polynomial<OutCont1>,
            &mut Polynomial<OutCont2>,
            &FourierPolynomial<AlignedVec<Complex64>>,
            &BackwardCorrector<&'static [Complex64]>,
        ),
    ) where
        Polynomial<OutCont1>: AsMutTensor<Element = Coef>,
        Polynomial<OutCont2>: AsMutTensor<Element = Coef>,
        FourierPolynomial<InCont1>: AsMutTensor<Element = Complex64>,
        FourierPolynomial<InCont2>: AsMutTensor<Element = Complex64>,
    {
        ck_dim_eq!(self.polynomial_size().0 =>
            fourier_poly_1.polynomial_size().0,
            poly_1.polynomial_size().0,
            fourier_poly_2.polynomial_size().0,
            poly_2.polynomial_size().0
        );

        // first we deal with the first root of unity
        let fp1 = fourier_poly_1.as_mut_tensor().as_mut_slice();
        let fp2 = fourier_poly_2.as_mut_tensor().as_mut_slice();
        fp1[0] = Complex64::new(fp1[0].re - fp2[0].im, fp1[0].im + fp2[0].re);
        fp1[1] = Complex64::new(fp1[1].re - fp2[1].im, fp1[1].im + fp2[1].re);

        let mut first_view = fourier_poly_1.as_mut_tensor().as_mut_slice();
        let (a_first, a_second) = split_in_mut_mut(&mut first_view, self.polynomial_size().0);

        for (fft_aj, (rot_fft_aj, fft_bj)) in zip!(
            a_first.iter_mut(),
            a_second.iter_mut().rev(),
            fourier_poly_2.as_mut_tensor().as_mut_slice()[2..].iter()
        ) {
            let re = fft_aj.re;
            let im = fft_aj.im;
            *fft_aj = Complex64::new(fft_aj.re - fft_bj.im, fft_aj.im + fft_bj.re);
            *rot_fft_aj = Complex64::new(re + fft_bj.im, -im + fft_bj.re);
        }

        // We perform the backward fft
        self.backward_plan
            .c2c(
                &mut fourier_poly_1.as_mut_tensor().as_mut_slice(),
                &mut self.temporary.as_mut_tensor().as_mut_slice(),
            )
            .expect("put_in_coeff_domain: fft.c2c threw an error...");

        convert_function(poly_1, poly_2, &self.temporary, &self.correctors.backward)
    }
}

fn split_in_mut_imut(sli: &mut [Complex64], big_n: usize) -> (&mut [Complex64], &[Complex64]) {
    let len = sli.len() - 2;
    let mid = big_n / 2 - 1;
    let ptr = unsafe { sli.as_mut_ptr().add(2) };
    unsafe {
        assert!(mid <= len);

        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts(ptr.add(mid), len - mid),
        )
    }
}

fn split_in_imut_mut(sli: &mut [Complex64], big_n: usize) -> (&[c64], &mut [c64]) {
    let len = sli.len() - 2;
    let mid = big_n / 2 - 1;
    let ptr = unsafe { sli.as_mut_ptr().add(2) };
    unsafe {
        assert!(mid <= len);

        (
            slice::from_raw_parts(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn split_in_mut_mut(s: &mut [Complex64], big_n: usize) -> (&mut [Complex64], &mut [Complex64]) {
    let len = s.len() - 2;
    let mid = big_n / 2 - 1;
    let ptr = unsafe { s.as_mut_ptr().add(2) };
    unsafe {
        assert!(mid <= len);
        (
            slice::from_raw_parts_mut(ptr, mid),
            slice::from_raw_parts_mut(ptr.add(mid), len - mid),
        )
    }
}

fn replicate_coefficients(fft_a: &mut [Complex64], fft_b: &mut [Complex64], big_n: usize) {
    // in fft_a there is FFT(coeff_a + i coeff_b) we now extract
    // the fourier transfform of coeff_a and of coeff_b using the fact
    // that halves of the roots of -1 are conjugate to the other half
    fft_b[0] = fft_a[1];
    fft_b[1] = fft_a[0];

    let mut tmp: Complex64;
    let s = Complex64::new(0., -0.5);
    tmp = fft_a[0];
    fft_a[0] = (fft_a[0] + fft_b[0].conj()) * 0.5;
    fft_b[0] = (tmp - fft_b[0].conj()) * s;
    tmp = fft_a[1];
    fft_a[1] = (fft_a[1] + fft_b[1].conj()) * 0.5;
    tmp -= fft_b[1].conj();
    fft_b[1] = Complex64::new(tmp.im / 2., -tmp.re / 2.);

    let (first_part, second_part) = split_in_mut_imut(fft_a, big_n);

    for (x_i, (x_rot_i, y_i)) in zip!(
        first_part.iter_mut(),
        second_part.iter().rev(),
        fft_b[2..].iter_mut()
    ) {
        tmp = *x_i;
        *x_i = (*x_i + x_rot_i.conj()) * 0.5;
        tmp -= x_rot_i.conj();
        *y_i = Complex64::new(tmp.im / 2., -tmp.re / 2.);
    }
}

fn regular_convert_forward_single_torus<InCont, Coef>(
    out: &mut FourierPolynomial<AlignedVec<Complex64>>,
    inp: &Polynomial<InCont>,
    corr: &ForwardCorrector<&'static [Complex64]>,
) where
    Polynomial<InCont>: AsRefTensor<Element = Coef>,
    Coef: UnsignedTorus,
{
    ck_dim_eq!(inp.as_tensor().len() => corr.as_tensor().len(), out.as_tensor().len());
    for (input, (corrector, output)) in inp
        .as_tensor()
        .iter()
        .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
    {
        *output = Complex64::new(input.into_torus(), 0.) * corrector;
    }
}

fn regular_convert_forward_two_torus<InCont1, InCont2, Coef>(
    out: &mut FourierPolynomial<AlignedVec<Complex64>>,
    inp1: &Polynomial<InCont1>,
    inp2: &Polynomial<InCont2>,
    corr: &ForwardCorrector<&'static [Complex64]>,
) where
    Polynomial<InCont1>: AsRefTensor<Element = Coef>,
    Polynomial<InCont2>: AsRefTensor<Element = Coef>,
    Coef: UnsignedTorus,
{
    ck_dim_eq!(
        inp1.as_tensor().len() =>
        corr.as_tensor().len(),
        out.as_tensor().len(),
        inp2.as_tensor().len()
    );
    for (input_1, (input_2, (corrector, output))) in inp1.as_tensor().iter().zip(
        inp2.as_tensor()
            .iter()
            .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut())),
    ) {
        *output = Complex64::new(input_1.into_torus(), input_2.into_torus()) * corrector;
    }
}

fn regular_convert_forward_single_integer<InCont, Coef>(
    out: &mut FourierPolynomial<AlignedVec<Complex64>>,
    inp: &Polynomial<InCont>,
    corr: &ForwardCorrector<&'static [Complex64]>,
) where
    Polynomial<InCont>: AsRefTensor<Element = Coef>,
    Coef: UnsignedInteger,
{
    ck_dim_eq!(inp.as_tensor().len() => corr.as_tensor().len(), out.as_tensor().len());
    for (input, (corrector, output)) in inp
        .as_tensor()
        .iter()
        .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
    {
        let val: f64 = (*input).into_signed().cast_into();
        *output = Complex64::new(val, 0.) * corrector;
    }
}

fn regular_convert_forward_two_integer<InCont1, InCont2, Coef>(
    out: &mut FourierPolynomial<AlignedVec<Complex64>>,
    inp1: &Polynomial<InCont1>,
    inp2: &Polynomial<InCont2>,
    corr: &ForwardCorrector<&'static [Complex64]>,
) where
    Polynomial<InCont1>: AsRefTensor<Element = Coef>,
    Polynomial<InCont2>: AsRefTensor<Element = Coef>,
    Coef: UnsignedInteger,
{
    ck_dim_eq!(
        inp1.as_tensor().len() =>
        corr.as_tensor().len(),
        out.as_tensor().len(),
        inp2.as_tensor().len()
    );
    for (input_1, (input_2, (corrector, output))) in inp1.as_tensor().iter().zip(
        inp2.as_tensor()
            .iter()
            .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut())),
    ) {
        let re: f64 = (*input_1).into_signed().cast_into();
        let im: f64 = (*input_2).into_signed().cast_into();
        *output = Complex64::new(re, im) * corrector;
    }
}

fn regular_convert_add_backward_single_torus<OutCont, Coef>(
    out: &mut Polynomial<OutCont>,
    inp: &FourierPolynomial<AlignedVec<Complex64>>,
    corr: &BackwardCorrector<&'static [Complex64]>,
) where
    Polynomial<OutCont>: AsMutTensor<Element = Coef>,
    Coef: UnsignedTorus,
{
    ck_dim_eq!(inp.as_tensor().len() => corr.as_tensor().len(), out.as_tensor().len());
    for (input, (corrector, output)) in inp
        .as_tensor()
        .iter()
        .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
    {
        let interm = (input * corrector).re;
        *output = output.wrapping_add(Coef::from_torus(interm));
    }
}

fn regular_convert_add_backward_single_integer<OutCont, Coef>(
    out: &mut Polynomial<OutCont>,
    inp: &FourierPolynomial<AlignedVec<Complex64>>,
    corr: &BackwardCorrector<&'static [Complex64]>,
) where
    Polynomial<OutCont>: AsMutTensor<Element = Coef>,
    Coef: UnsignedInteger,
{
    ck_dim_eq!(inp.as_tensor().len() => corr.as_tensor().len(), out.as_tensor().len());
    for (input, (corrector, output)) in inp
        .as_tensor()
        .iter()
        .zip(corr.as_tensor().iter().zip(out.as_mut_tensor().iter_mut()))
    {
        let interm = (input * corrector).re;
        let out: Coef::Signed = interm.round().cast_into();
        *output = output.wrapping_add(out.into_unsigned());
    }
}

fn regular_convert_add_backward_two_torus<OutCont1, OutCont2, Coef>(
    out1: &mut Polynomial<OutCont1>,
    out2: &mut Polynomial<OutCont2>,
    inp: &FourierPolynomial<AlignedVec<Complex64>>,
    corr: &BackwardCorrector<&'static [Complex64]>,
) where
    Polynomial<OutCont1>: AsMutTensor<Element = Coef>,
    Polynomial<OutCont2>: AsMutTensor<Element = Coef>,
    Coef: UnsignedTorus,
{
    ck_dim_eq!(
        out1.as_tensor().len() =>
        corr.as_tensor().len(),
        inp.as_tensor().len(),
        out2.as_tensor().len()
    );
    for (output_1, (output_2, (corrector, input))) in out1.as_mut_tensor().iter_mut().zip(
        out2.as_mut_tensor()
            .iter_mut()
            .zip(corr.as_tensor().iter().zip(inp.as_tensor().iter())),
    ) {
        let interm = input * corrector;
        let re_interm = interm.re;
        let im_interm = interm.im;
        *output_1 = output_1.wrapping_add(Coef::from_torus(re_interm));
        *output_2 = output_2.wrapping_add(Coef::from_torus(im_interm));
    }
}

fn regular_convert_add_backward_two_integer<OutCont1, OutCont2, Coef>(
    out1: &mut Polynomial<OutCont1>,
    out2: &mut Polynomial<OutCont2>,
    inp: &FourierPolynomial<AlignedVec<Complex64>>,
    corr: &BackwardCorrector<&'static [Complex64]>,
) where
    Polynomial<OutCont1>: AsMutTensor<Element = Coef>,
    Polynomial<OutCont2>: AsMutTensor<Element = Coef>,
    Coef: UnsignedInteger,
{
    ck_dim_eq!(
        out1.as_tensor().len() =>
        corr.as_tensor().len(),
        inp.as_tensor().len(),
        out2.as_tensor().len()
    );
    for (output_1, (output_2, (corrector, input))) in out1.as_mut_tensor().iter_mut().zip(
        out2.as_mut_tensor()
            .iter_mut()
            .zip(corr.as_tensor().iter().zip(inp.as_tensor().iter())),
    ) {
        let interm = input * corrector;
        let out_1: Coef::Signed = interm.re.round().cast_into();
        let out_2: Coef::Signed = interm.im.round().cast_into();
        *output_1 = output_1.wrapping_add(out_1.into_unsigned());
        *output_2 = output_2.wrapping_add(out_2.into_unsigned());
    }
}
