//! FFT Related Operations
//! * Contains functions dealing with FFT and its different implementations.

#[cfg(test)]
mod tests;

use crate::types::{C2CPlanTorus, CTorus, FTorus};
use crate::Types;
use fftw::array::AlignedVec;
use fftw::plan::*;
use itertools::izip;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[allow(unused_imports)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[allow(unused_imports)]
use std::mem::transmute;
use std::slice;

pub trait FFT: Sized {
    fn reconstruct_half(slice: &mut [CTorus], big_n: usize);

    fn split_in_mut_mut(sli: &mut [CTorus], big_n: usize) -> (&mut [CTorus], &mut [CTorus]);

    fn split_in_imut_mut(sli: &mut [CTorus], big_n: usize) -> (&[CTorus], &mut [CTorus]);

    fn split_in_mut_imut(sli: &mut [CTorus], big_n: usize) -> (&mut [CTorus], &[CTorus]);

    fn put_in_fft_domain_storus(
        fft_b: &mut AlignedVec<CTorus>,
        tmp: &mut AlignedVec<CTorus>,
        coeff_b: &[Self],
        twiddles: &[CTorus],
        fft: &mut C2CPlanTorus,
    );
    fn put_in_fft_domain_torus(
        fft_b: &mut AlignedVec<CTorus>,
        tmp: &mut AlignedVec<CTorus>,
        coeff_b: &[Self],
        twiddles: &[CTorus],
        fft: &mut C2CPlanTorus,
    );

    fn put_2_in_fft_domain_torus(
        fft_a: &mut AlignedVec<CTorus>,
        fft_b: &mut AlignedVec<CTorus>,
        coeff_a: &[Self],
        coeff_b: &[Self],
        twiddles: &[CTorus],
        fft: &mut C2CPlanTorus,
    );
    fn put_2_in_fft_domain_storus(
        fft_a: &mut AlignedVec<CTorus>,
        fft_b: &mut AlignedVec<CTorus>,
        coeff_a: &[Self],
        coeff_b: &[Self],
        twiddles: &[CTorus],
        fft: &mut C2CPlanTorus,
    );
    fn put_in_coeff_domain(
        coeff_b: &mut [Self],
        tmp: &mut AlignedVec<CTorus>,
        fft_b: &mut AlignedVec<CTorus>,
        inverse_twiddles: &[CTorus],
        ifft: &mut C2CPlanTorus,
    );
    fn put_2_in_coeff_domain(
        coeff_a: &mut [Self],
        coeff_b: &mut [Self],
        fft_a: &mut AlignedVec<CTorus>,
        fft_b: &mut AlignedVec<CTorus>,
        inverse_twiddles: &[CTorus],
        ifft: &mut C2CPlanTorus,
    );

    fn schoolbook_multiply(p_res: &mut [Self], p_tor: &[Self], p_int: &[Self], degree: usize);

    fn compute_monomial_polynomial_prod_torus(
        degree: usize,
        p_tor: &[Self],
        p_int: &[Self],
        monomial: usize,
    ) -> Self;
}

macro_rules! impl_trait_fft {
    ($T:ty,$DOC:expr) => {
        impl FFT for $T {
            /// When dealing with real polynomial of degree big_n - 1, we only need to store
            ///  half of its fourier transform as one half is the conjuagate of the other half
            /// This function reconstruct a vector of size big_n using big_n / 2 values and
            /// the conjugate relationship.
            /// # Arguments
            /// * `slice` - fourier representation of a real polynomial
            /// * `big_n` - number of coefficients of the polynomial e.g. degree + 1
            fn reconstruct_half(slice: &mut [CTorus], big_n: usize) {
                let (first, second) = Self::split_in_imut_mut(slice, big_n);

                for (a, conja) in first.iter().rev().zip(second.iter_mut()) {
                    *conja = a.conj();
                }
            }

            /// Split a slice of CTorus elements in two mutable references
            /// # Comments
            /// * This function uses unsafe slice::from_raw_parts_mut and as_mut_ptr
            /// # Arguments
            /// * `s` - slice of CTorus elements to be split
            /// * `big_n` - length of the slice s
            /// # Example
            /// ```rust
            /// use concrete::core_api::math::FFT;
            /// use concrete::types::CTorus;
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024 ;
            /// let mut c: Vec<CTorus> = vec![CTorus::new(0., 0.); big_n] ;
            ///
            /// let (first, second) = <Torus as FFT>::split_in_mut_mut(&mut c, big_n) ;
            /// ```
            fn split_in_mut_mut(
                s: &mut [CTorus],
                big_n: usize,
            ) -> (&mut [CTorus], &mut [CTorus]) {
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

            /// Split a slice of CTorus elements in a immutable reference and a mutable reference
            /// # Comments
            /// * This function uses unsafe slice::from_raw_parts_mut and as_mut_ptr
            /// # Arguments
            /// * `s` - slice of CTorus elements to be split
            /// * `big_n` - length of the slice s
            /// # Example
            /// ```rust
            /// use concrete::core_api::math::FFT;
            /// use concrete::types::CTorus;
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024 ;
            /// let mut c: Vec<CTorus> = vec![CTorus::new(0., 0.); big_n] ;
            ///
            /// let (first, second) = <Torus as FFT>::split_in_imut_mut(&mut c, big_n) ;
            /// ```
            fn split_in_imut_mut(sli: &mut [CTorus], big_n: usize) -> (&[CTorus], &mut [CTorus]) {
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

            /// Split a slice of CTorus elements in a mutable reference and a immutable reference
            /// # Comments
            /// * This function uses unsafe slice::from_raw_parts_mut and as_mut_ptr
            /// # Arguments
            /// * `s` - slice of CTorus elements to be split
            /// * `big_n` - length of the slice s
            /// # Example
            /// ```rust
            /// use concrete::core_api::math::FFT;
            /// use concrete::types::CTorus;
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024 ;
            /// let mut c: Vec<CTorus> = vec![CTorus::new(0., 0.); big_n] ;
            ///
            /// let (first, second) = <Torus as FFT>::split_in_mut_imut(&mut c, big_n) ;
            /// ```
            fn split_in_mut_imut(sli: &mut [CTorus], big_n: usize) -> (&mut [CTorus], &[CTorus]) {
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
            /// Put a Torus polynomial into the FFT using fftw's c2C algorithm
            /// # Arguments
            /// * `fft_b` - Aligned Vector of FFT(coeff_b) (output)
            /// * `tmp` - Aligned Vector used as temporary variable
            /// * `coeff_b` - Torus slice representing a signed polynomial with Torus coefficients
            /// * `twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `fft` - FFTW Plan
            /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
            ///
            /// // Create and fill a vector with Torus element
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_b) ;
            ///
            /// FFT::put_in_fft_domain_storus(
            ///     &mut fft_b,
            ///     &mut tmp,
            ///     &coeff_b,
            ///     TWIDDLES_TORUS_external!(big_n),
            ///     &mut fft,
            /// );
            /// ```
            fn put_in_fft_domain_storus(
                fft_b: &mut AlignedVec<CTorus>,
                tmp: &mut AlignedVec<CTorus>,
                coeff_b: &[$T],
                twiddles: &[CTorus],
                fft: &mut C2CPlanTorus,
            ) {
                debug_assert!(fft_b.len() == tmp.len(), "fft_b size != tmp size");
                debug_assert!(fft_b.len() == coeff_b.len(), "fft_b size != coeff_b size");
                debug_assert!(fft_b.len() == twiddles.len(), "fft_b size != twiddles size");

                for (ref_tmp, coeff_b_val, twiddle) in
                    izip!(tmp.iter_mut(), coeff_b.iter(), twiddles.iter())
                {
                    *ref_tmp = CTorus::new((*coeff_b_val as <$T as Types>::STorus) as FTorus, 0.)
                        * twiddle;
                }
                fft.c2c(tmp, fft_b)
                    .expect("put_in_fft_domain: fft.c2c threw an error...");
            }

            /// Put a Torus polynomial into the FFT using fftw's c2C algorithm
            /// # Arguments
            /// * `fft_b` - Aligned Vector of FFT(coeff_b) (output)
            /// * `tmp` - Aligned Vector used as temporary variable
            /// * `coeff_b` - Torus slice representing a unsigned polynomial with Torus coefficients
            /// * `twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `fft` - FFTW Plan
            /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
            ///
            /// // Create and fill a vector with Torus element
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_b) ;
            ///
            /// FFT::put_in_fft_domain_torus(
            ///     &mut fft_b,
            ///     &mut tmp,
            ///     &coeff_b,
            ///     TWIDDLES_TORUS_external!(big_n),
            ///     &mut fft,
            /// );
            /// ```
            fn put_in_fft_domain_torus(
                fft_b: &mut AlignedVec<CTorus>,
                tmp: &mut AlignedVec<CTorus>,
                coeff_b: &[$T],
                twiddles: &[CTorus],
                fft: &mut C2CPlanTorus,
            ) {
                debug_assert!(fft_b.len() == tmp.len(), "fft_b size != tmp size");
                debug_assert!(fft_b.len() == coeff_b.len(), "fft_b size != coeff_b size");
                debug_assert!(fft_b.len() == twiddles.len(), "fft_b size != twiddles size");

                for (ref_tmp, coeff_b_val, twiddle) in
                    izip!(tmp.iter_mut(), coeff_b.iter(), twiddles.iter())
                {
                    *ref_tmp = CTorus::new(
                        *coeff_b_val as FTorus
                            * FTorus::powi(2., -(<$T as Types>::TORUS_BIT as i32)),
                        0.,
                    ) * twiddle;
                }
                fft.c2c(tmp, fft_b)
                    .expect("put_in_fft_domain: fft.c2c threw an error...");
            }

            /// Put two Torus polynomials into the FFT using fftw's c2C algorithm
            /// # Arguments
            /// * `fft_a` - Aligned Vector of FFT(coeff_a) (output)
            /// * `fft_b` - Aligned Vector of FFT(coeff_b) (output)
            /// * `coeff_a` - Torus slice representing a polynomial
            /// * `coeff_b` - Torus slice representing a polynomial
            /// * `twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `fft` - FFTW Plan
            /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
            ///
            /// // Create and fill a vector with Torus element
            /// let mut coeff_a: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_a) ;
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_b) ;
            ///
            /// FFT::put_2_in_fft_domain_torus(
            ///     &mut fft_a,
            ///     &mut fft_b,
            ///     &coeff_a,
            ///     &coeff_b,
            ///     TWIDDLES_TORUS_external!(big_n),
            ///     &mut fft,
            /// );
            ///
            fn put_2_in_fft_domain_torus(
                fft_a: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                coeff_a: &[$T],
                coeff_b: &[$T],
                twiddles: &[CTorus],
                fft: &mut C2CPlanTorus,
            ) {
                debug_assert!(
                    coeff_a.len() == coeff_b.len(),
                    "coeff_a size != coeff_b size"
                );
                debug_assert!(fft_b.len() == twiddles.len(), "fft_b size != TWIDDLES size");
                debug_assert!(coeff_a.len() == fft_b.len(), "coeff_a size != fft_b size");
                // we create a CTorus polynomial with coefficient of A in the real part
                // and coeff of B in the imaginary part
                for (fft_b_i, coeff_a_i, coeff_b_i, twiddle) in izip!(
                    fft_b.iter_mut(),
                    coeff_a.iter(),
                    coeff_b.iter(),
                    twiddles.iter()
                ) {
                    *fft_b_i = CTorus::new(
                        *coeff_a_i as FTorus * FTorus::powi(2., -(<$T as Types>::TORUS_BIT as i32)),
                        *coeff_b_i as FTorus * FTorus::powi(2., -(<$T as Types>::TORUS_BIT as i32)),
                    ) * twiddle;
                }
                // doing the actual fourier transform
                fft.c2c(fft_b, fft_a)
                    .expect("put_2_in_fft_domain: fft.c2c threw an error...");
                // in fft_a there is FFT(coeff_a + i coeff_b) we now extract
                // the fourier transfform of coeff_a and of coeff_b using the fact
                // that halves of the roots of -1 are conjugate to the other half
                debug_assert!(
                    fft_a.len() == fft_b.len(),
                    "fft_a size = {} != fft_b size = {}",
                    fft_a.len(),
                    fft_b.len()
                );

                // First we deal with the two first elements which contain the evaluation
                // of the polynomial of the first root of unity and its conjuguate.
                fft_b[0] = fft_a[1];
                fft_b[1] = fft_a[0];
                let mut tmp: CTorus;
                let s = CTorus::new(0., -0.5);
                tmp = fft_a[0];
                fft_a[0] = (fft_a[0] + fft_b[0].conj()) * 0.5;
                fft_b[0] = (tmp - fft_b[0].conj()) * s;
                tmp = fft_a[1];
                fft_a[1] = (fft_a[1] + fft_b[1].conj()) * 0.5;
                tmp -= fft_b[1].conj();
                fft_b[1] = CTorus::new(tmp.im / 2., -tmp.re / 2.);

                let (first_part, second_part) = Self::split_in_mut_imut(fft_a, coeff_a.len());

                for (x_i, x_rot_i, y_i) in izip!(
                    first_part.iter_mut(),
                    second_part.iter().rev(),
                    fft_b[2..].iter_mut()
                ) {
                    tmp = *x_i;
                    *x_i = (*x_i + x_rot_i.conj()) * 0.5;
                    tmp -= x_rot_i.conj();
                    *y_i = CTorus::new(tmp.im / 2., -tmp.re / 2.);
                }

            }

            /// Put two Torus polynomials into the FFT using fftw's c2C algorithm
            /// # Arguments
            /// * `fft_a` - Aligned Vector of FFT(coeff_a) (output)
            /// * `fft_b` - Aligned Vector of FFT(coeff_b) (output)
            /// * `coeff_a` - Torus slice representing a polynomial
            /// * `coeff_b` - Torus slice representing a polynomial
            /// * `twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `fft` - FFTW Plan
            /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
            ///
            /// // Create and fill a vector with Torus element
            /// let mut coeff_a: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_a) ;
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_b) ;
            ///
            /// FFT::put_2_in_fft_domain_storus(
            ///     &mut fft_a,
            ///     &mut fft_b,
            ///     &coeff_a,
            ///     &coeff_b,
            ///     TWIDDLES_TORUS_external!(big_n),
            ///     &mut fft,
            /// );
            ///
            #[cfg(not(target_feature = "avx2"))]
            fn put_2_in_fft_domain_storus(
                fft_a: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                coeff_a: &[$T],
                coeff_b: &[$T],
                twiddles: &[CTorus],
                fft: &mut C2CPlanTorus,
            ) {
                debug_assert!(
                    coeff_a.len() == coeff_b.len(),
                    "coeff_a size != coeff_b size"
                );
                debug_assert!(fft_b.len() == twiddles.len(), "fft_b size != TWIDDLES size");
                debug_assert!(coeff_a.len() == fft_b.len(), "coeff_a size != fft_b size");
                // we create a CTorus polynomial with coefficient of A in the real part
                // and coeff of B in the imaginary part
                for (fft_b_i, coeff_a_i, coeff_b_i, twiddle) in izip!(
                    fft_b.iter_mut(),
                    coeff_a.iter(),
                    coeff_b.iter(),
                    twiddles.iter()
                ) {
                    *fft_b_i = CTorus::new(
                        (*coeff_a_i as <$T as Types>::STorus) as FTorus,
                        (*coeff_b_i as <$T as Types>::STorus) as FTorus,
                    ) * twiddle;
                }
                // doing the actual fourier transform
                fft.c2c(fft_b, fft_a)
                    .expect("put_2_in_fft_domain: fft.c2c threw an error...");
                // in fft_a there is FFT(coeff_a + i coeff_b) we now extract
                // the fourier transfform of coeff_a and of coeff_b using the fact
                // that halves of the roots of -1 are conjugate to the other half
                debug_assert!(
                    fft_a.len() == fft_b.len(),
                    "fft_a size = {} != fft_b size = {}",
                    fft_a.len(),
                    fft_b.len()
                );

                fft_b[0] = fft_a[1];
                fft_b[1] = fft_a[0];

                let mut tmp: CTorus;
                let s = CTorus::new(0., -0.5);
                tmp = fft_a[0];
                fft_a[0] = (fft_a[0] + fft_b[0].conj()) * 0.5;
                fft_b[0] = (tmp - fft_b[0].conj()) * s;
                tmp = fft_a[1];
                fft_a[1] = (fft_a[1] + fft_b[1].conj()) * 0.5;
                tmp -= fft_b[1].conj();
                fft_b[1] = CTorus::new(tmp.im / 2., -tmp.re / 2.);

                let (first_part, second_part) = Self::split_in_mut_imut(fft_a, coeff_a.len());

                for (x_i, x_rot_i, y_i) in izip!(
                    first_part.iter_mut(),
                    second_part.iter().rev(),
                    fft_b[2..].iter_mut()
                ) {
                    tmp = *x_i;
                    *x_i = (*x_i + x_rot_i.conj()) * 0.5;
                    tmp -= x_rot_i.conj();
                    *y_i = CTorus::new(tmp.im / 2., -tmp.re / 2.);
                }

            }

            #[cfg(target_feature = "avx2")]
            /// Put two Torus polynomials into the FFT using fftw's c2C algorithm
            /// # Arguments
            /// * `fft_a` - Aligned Vector of FFT(coeff_a) (output)
            /// * `fft_b` - Aligned Vector of FFT(coeff_b) (output)
            /// * `coeff_a` - Torus slice representing a signed polynomial with Torus coefficients
            /// * `coeff_b` - Torus slice representing a signed polynomial with Torus coefficients
            /// * `twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `fft` - FFTW Plan
            /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Forward, Flag::Measure).unwrap();
            ///
            /// // Create and fill a vector with Torus element
            /// let mut coeff_a: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_a) ;
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            /// Tensor::uniform_random_default(&mut coeff_b) ;
            ///
            /// FFT::put_2_in_fft_domain_storus(
            ///     &mut fft_a,
            ///     &mut fft_b,
            ///     &coeff_a,
            ///     &coeff_b,
            ///     TWIDDLES_TORUS_external!(big_n),
            ///     &mut fft,
            /// );
            ///
            fn put_2_in_fft_domain_storus(
                fft_a: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                coeff_a: &[$T],
                coeff_b: &[$T],
                twiddles: &[CTorus],
                fft: &mut C2CPlanTorus,
            ) {
                debug_assert!(
                    coeff_a.len() == coeff_b.len(),
                    "coeff_a size != coeff_b size"
                );
                debug_assert!(fft_b.len() == twiddles.len(), "fft_b size != TWIDDLES size");
                debug_assert!(coeff_a.len() == fft_b.len(), "coeff_a size != fft_b size");
                // we create a CTorus polynomial with coefficient of A in the real part
                // and coeff of B in the imaginary part
                // We do the multiplication with the twiddle factors with avx/sse
                for (fft_b_i, coeff_a_i, coeff_b_i, vec_tw) in izip!(
                    fft_b.chunks_mut(2),
                    coeff_a.chunks(2),
                    coeff_b.chunks(2),
                    twiddles.chunks(2)
                ) {
                    unsafe {
                    let vec_b: __m256d = _mm256_setr_pd(
                        (*coeff_a_i.get_unchecked(0) as <$T as Types>::STorus) as FTorus,
                        (*coeff_b_i.get_unchecked(0) as <$T as Types>::STorus) as FTorus,
                        (*coeff_a_i.get_unchecked(1) as <$T as Types>::STorus) as FTorus,
                        (*coeff_b_i.get_unchecked(1) as <$T as Types>::STorus) as FTorus,
                    );
                    let vec_tw: __m256d = _mm256_setr_pd(
                        vec_tw.get_unchecked(0).re ,
                        vec_tw.get_unchecked(0).im ,
                        vec_tw.get_unchecked(1).re ,
                        vec_tw.get_unchecked(1).im ,
                    ) ;
                    let neg: __m256d = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

                    // Multiply
                    let vec_btw: __m256d = _mm256_mul_pd(vec_b, vec_tw);
                    // Permute the real and imaginary elements
                    let mut vec_tw = _mm256_permute_pd(vec_tw, 0x5);
                    // Negate the imaginary elements of rlwe
                    vec_tw = _mm256_mul_pd(vec_tw, neg);
                    // Multiply
                    let vec_btw_bis: __m256d = _mm256_mul_pd(vec_b, vec_tw);
                    // Horizontal substract  and add
                    let interm:(f64, f64, f64, f64) = transmute(
                        _mm256_hsub_pd(vec_btw, vec_btw_bis));

                        fft_b_i.get_unchecked_mut(0).re = interm.0 ;
                        fft_b_i.get_unchecked_mut(0).im = interm.1 ;
                        fft_b_i.get_unchecked_mut(1).re = interm.2 ;
                        fft_b_i.get_unchecked_mut(1).im = interm.3 ;

                    }
                }
                // doing the actual fourier transform
                fft.c2c(fft_b, fft_a)
                    .expect("put_2_in_fft_domain: fft.c2c threw an error...");
                // in fft_a there is FFT(coeff_a + i coeff_b) we now extract
                // the fourier transfform of coeff_a and of coeff_b using the fact
                // that halves of the roots of -1 are conjugate to the other half
                debug_assert!(
                    fft_a.len() == fft_b.len(),
                    "fft_a size = {} != fft_b size = {}",
                    fft_a.len(),
                    fft_b.len()
                );

                fft_b[0] = fft_a[1];
                fft_b[1] = fft_a[0];
                let mut tmp: CTorus;
                let s = CTorus::new(0., -0.5);
                tmp = fft_a[0];
                fft_a[0] = (fft_a[0] + fft_b[0].conj()) * 0.5;
                fft_b[0] = (tmp - fft_b[0].conj()) * s;
                tmp = fft_a[1];
                fft_a[1] = (fft_a[1] + fft_b[1].conj()) * 0.5;
                tmp = tmp - fft_b[1].conj();
                fft_b[1] = CTorus::new(tmp.im / 2., -tmp.re / 2.);

                let (first_part, second_part) = Self::split_in_mut_imut(fft_a, coeff_a.len());

                for (x_i, x_rot_i, y_i) in izip!(
                    first_part.iter_mut(),
                    second_part.iter().rev(),
                    fft_b[2..].iter_mut()
                ) {
                    tmp = *x_i;
                    *x_i = (*x_i + x_rot_i.conj()) * 0.5;
                    tmp = tmp - x_rot_i.conj();
                    *y_i = CTorus::new(tmp.im / 2., -tmp.re / 2.);
                }
            }

            /// Perform the IFFT using fftw's c2C algorithm
            /// # Arguments
            /// * `coeff_b` - Torus slice representing a polynomial (output)
            /// * `tmp` - Aligned Vector used as temporary variable
            /// * `fft_b` - Aligned Vector of FFT(coeff_b)
            /// * `inverse_twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `ifft` - FFTW Plan for backward FFT
             /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::INVERSE_TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut tmp: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Backward, Flag::Measure).unwrap();
            ///
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            ///
            /// FFT::put_in_coeff_domain(
            ///     &mut coeff_b,
            ///     &mut tmp,
            ///     &mut fft_b,
            ///     INVERSE_TWIDDLES_TORUS_external!(big_n),
            ///     &mut ifft,
            /// );
            /// ```
            fn put_in_coeff_domain(
                coeff_b: &mut [$T],
                tmp: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                inverse_twiddles: &[CTorus],
                ifft: &mut C2CPlanTorus,
            ) {
                debug_assert!(coeff_b.len() == tmp.len(), "coeff_b size != tmp size");
                debug_assert!(coeff_b.len() == fft_b.len(), "coeff_b size != fft_b size");
                debug_assert!(
                    coeff_b.len() == inverse_twiddles.len(),
                    "coeff_b size != inverse_twiddles size"
                );


                // first we deal with the first root of unity
                let (b_first, b_second) = Self::split_in_imut_mut(fft_b, coeff_b.len());

                for (fft_bj, rot_fft_bj) in izip!(
                    b_first.iter(),
                    b_second.iter_mut().rev(),
                ) {
                    *rot_fft_bj = fft_bj.conj();
                }

                ifft.c2c(fft_b, tmp)
                    .expect("put_in_coeff_domain: fft.c2c threw an error...");

                for (coeff_b_ref, coeff_f_b_0, twiddle) in
                    izip!(coeff_b.iter_mut(), tmp.iter(), inverse_twiddles.iter())
                {
                    let interm = (coeff_f_b_0 * twiddle).re;
                    let mut y = (interm - FTorus::floor(interm))
                    * FTorus::powi(2., <$T as Types>::TORUS_BIT as i32) ;
                    let carry = y - FTorus::floor(y) ;
                    if carry >=0.5 {y += 1.;}
                    *coeff_b_ref = coeff_b_ref.wrapping_add(y as $T);

                }
            }

            #[cfg(not(target_feature = "avx2"))]
            /// Perform the IFFT of two polynomials using fftw's c2C algorithm
            /// # Arguments
            /// * `coeff_a` - Torus slice representing a polynomial (output)
            /// * `coeff_b` - Torus slice representing a polynomial (output)
            /// * `fft_a` - Aligned Vector of FFT(coeff_a)
            /// * `fft_b` - Aligned Vector of FFT(coeff_b)
            /// * `inverse_twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `ifft` - FFTW Plan for backward FFT
             /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::INVERSE_TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Backward, Flag::Measure).unwrap();
            ///
            /// let mut coeff_a: Vec<Torus> = vec![0; big_n];
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            ///
            /// FFT::put_2_in_coeff_domain(
            ///     &mut coeff_a,
            ///     &mut coeff_b,
            ///     &mut fft_a,
            ///     &mut fft_b,
            ///     INVERSE_TWIDDLES_TORUS_external!(big_n),
            ///     &mut ifft,
            /// );
            /// ```
            fn put_2_in_coeff_domain(
                coeff_a: &mut [$T],
                coeff_b: &mut [$T],
                fft_a: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                inverse_twiddles: &[CTorus],
                ifft: &mut C2CPlanTorus,
            ) {
                debug_assert!(
                    coeff_a.len() == coeff_b.len(),
                    "coeff_a size != coeff_b size"
                );
                debug_assert!(coeff_a.len() == fft_b.len(), "coeff_a size != fft_b size");

                // first we deal with the first root of unity
                fft_a[0] = CTorus::new(fft_a[0].re - fft_b[0].im, fft_a[0].im + fft_b[0].re);
                fft_a[1] = CTorus::new(fft_a[1].re - fft_b[1].im, fft_a[1].im + fft_b[1].re);
                let (a_first, a_second) = Self::split_in_mut_mut(fft_a, coeff_a.len());

                for (fft_aj, rot_fft_aj, fft_bj) in izip!(
                    a_first.iter_mut(),
                    a_second.iter_mut().rev(),
                    fft_b[2..].iter()
                ) {
                    let re = fft_aj.re;
                    let im = fft_aj.im;
                    *fft_aj = CTorus::new(fft_aj.re - fft_bj.im, fft_aj.im + fft_bj.re);
                    *rot_fft_aj = CTorus::new(re + fft_bj.im, -im + fft_bj.re);
                }

                ifft.c2c(fft_a, fft_b)
                    .expect("put_2_in_coeff_domain: fft.c2c threw an error...");

                for (coeff_ai, (coeff_bi, (fft_bi, twiddle))) in coeff_a.iter_mut().zip(
                    coeff_b
                        .iter_mut()
                        .zip(fft_b.iter().zip(inverse_twiddles.iter())),
                ) {
                    let interm: CTorus = *fft_bi * twiddle;
                    let re_interm: FTorus = interm.re;
                    let im_interm: FTorus = interm.im;

                    let mut y = (re_interm - FTorus::floor(re_interm))
                    * FTorus::powi(2., <$T as Types>::TORUS_BIT as i32) ;
                    let carry = y - FTorus::floor(y) ;
                    if carry >=0.5 {y += 1.;}
                    *coeff_ai = coeff_ai.wrapping_add(y as $T);

                    y = (im_interm - FTorus::floor(im_interm))
                    * FTorus::powi(2., <$T as Types>::TORUS_BIT as i32);
                    let carry = y - FTorus::floor(y) ;
                    if carry >=0.5 {y += 1.;}

                    *coeff_bi = coeff_bi.wrapping_add(y as $T) ;
                }
            }

            #[cfg(target_feature = "avx2")]
            /// Perform the IFFT of two polynomials using fftw's c2C algorithm
            /// # Arguments
            /// * `coeff_a` - Torus slice representing a polynomial (output)
            /// * `coeff_b` - Torus slice representing a polynomial (output)
            /// * `fft_a` - Aligned Vector of FFT(coeff_a)
            /// * `fft_b` - Aligned Vector of FFT(coeff_b)
            /// * `inverse_twiddles` - CTorus slice containing correcting factor to work with an FFT on T\[X\] / (X^N-1) instead of T\[X\] /(X^N +1)
            /// * `ifft` - FFTW Plan for backward FFT
             /// # Example
            /// ```rust
            /// #[macro_use]
            /// use concrete;
            /// use fftw::array::AlignedVec;
            /// use fftw::types::{Flag, Sign};
            /// use fftw::plan::C2CPlan;
            /// use concrete::core_api::math::{Tensor, FFT} ;
            /// use concrete::types::{CTorus, C2CPlanTorus};
            /// use concrete::INVERSE_TWIDDLES_TORUS_external;
            ///
            #[doc = $DOC]
            ///
            /// let big_n: usize = 1024;
            /// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
            /// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
            ///
            /// // Creation of FFTW plan
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[big_n], Sign::Backward, Flag::Measure).unwrap();
            ///
            /// let mut coeff_a: Vec<Torus> = vec![0; big_n];
            /// let mut coeff_b: Vec<Torus> = vec![0; big_n];
            ///
            /// FFT::put_2_in_coeff_domain(
            ///     &mut coeff_a,
            ///     &mut coeff_b,
            ///     &mut fft_a,
            ///     &mut fft_b,
            ///     INVERSE_TWIDDLES_TORUS_external!(big_n),
            ///     &mut ifft,
            /// );
            /// ```
            fn put_2_in_coeff_domain(
                coeff_a: &mut [$T],
                coeff_b: &mut [$T],
                fft_a: &mut AlignedVec<CTorus>,
                fft_b: &mut AlignedVec<CTorus>,
                inverse_twiddles: &[CTorus],
                ifft: &mut C2CPlanTorus,
            ) {
                debug_assert!(
                    coeff_a.len() == coeff_b.len(),
                    "coeff_a size != coeff_b size"
                );
                debug_assert!(coeff_a.len() == fft_b.len(), "coeff_a size != fft_b size");

                // first we deal with the first root of unity
                fft_a[0] = CTorus::new(fft_a[0].re - fft_b[0].im, fft_a[0].im + fft_b[0].re);
                fft_a[1] = CTorus::new(fft_a[1].re - fft_b[1].im, fft_a[1].im + fft_b[1].re);
                let (a_first, a_second) = Self::split_in_mut_mut(fft_a, coeff_a.len());

                for (fft_aj, rot_fft_aj, fft_bj) in izip!(
                    a_first.iter_mut(),
                    a_second.iter_mut().rev(),
                    fft_b[2..].iter()
                ) {
                    let re = fft_aj.re;
                    let im = fft_aj.im;
                    *fft_aj = CTorus::new(fft_aj.re - fft_bj.im, fft_aj.im + fft_bj.re);
                    *rot_fft_aj = CTorus::new(re + fft_bj.im, -im + fft_bj.re);
                }

                ifft.c2c(fft_a, fft_b)
                    .expect("put_2_in_coeff_domain: fft.c2c threw an error...");
                    unsafe {

                    let pow2_32 = FTorus::powi(2., <$T as Types>::TORUS_BIT as i32) ;

                    let multiplier = _mm256_set_pd(pow2_32, pow2_32, pow2_32, pow2_32);

                for (coeff_ai, (coeff_bi, (fft_bi, twiddle))) in coeff_a.chunks_mut(2).zip(
                    coeff_b
                        .chunks_mut(2)
                        .zip(fft_b.chunks(2).zip(inverse_twiddles.chunks(2))),
                ) {

                    let vec_b: __m256d = _mm256_setr_pd(
                        fft_bi.get_unchecked(0).re,
                        fft_bi.get_unchecked(0).im,
                        fft_bi.get_unchecked(1).re,
                        fft_bi.get_unchecked(1).im,
                    );
                    let mut vec_tw: __m256d = _mm256_setr_pd(
                        twiddle.get_unchecked(0).re,
                        twiddle.get_unchecked(0).im,
                        twiddle.get_unchecked(1).re,
                        twiddle.get_unchecked(1).im,
                    );
                    let neg: __m256d = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

                    // Multiply
                    let vec_btw: __m256d = _mm256_mul_pd(vec_b, vec_tw);
                    // Permute the real and imaginary elements of rlwe
                    vec_tw = _mm256_permute_pd(vec_tw, 0x5);
                    // Negate the imaginary elements of rlwe
                    vec_tw = _mm256_mul_pd(vec_tw, neg);
                    // Multiply
                    let vec_btw_bis: __m256d = _mm256_mul_pd(vec_b, vec_tw);
                    // Horizontal substract  and add
                    let mut interm =
                        _mm256_hsub_pd(vec_btw, vec_btw_bis);

                    // now we can do ([x - floor(x)] * 2**32 ).round() as $T

                    interm = _mm256_sub_pd(interm, _mm256_round_pd(interm, 0x01)) ;
                    interm = _mm256_round_pd(_mm256_mul_pd(interm, multiplier), 0x00);
                    let interm: (f64, f64, f64, f64) = transmute(interm);
                    *coeff_ai.get_unchecked_mut(0) = coeff_ai.get_unchecked(0).wrapping_add(interm.0 as $T) ;
                    *coeff_bi.get_unchecked_mut(0) = coeff_bi.get_unchecked(0).wrapping_add(interm.1 as $T) ;
                    *coeff_ai.get_unchecked_mut(1) = coeff_ai.get_unchecked(1).wrapping_add(interm.2 as $T) ;
                    *coeff_bi.get_unchecked_mut(1) = coeff_bi.get_unchecked(1).wrapping_add(interm.3 as $T) ;
                    }
                }
            }


            /// stores in p_res (torus polynomial) the result of p_tor * p_int (without FFT, quotient X^big_n + 1)
            /// p_tor : torus polynomial
            /// p_int : integer polynomial
            fn schoolbook_multiply(p_res: &mut [$T], p_tor: &[$T], p_int: &[$T], degree: usize) {
                for (i, monomial_i) in p_res.iter_mut().enumerate() {
                    *monomial_i =
                        Self::compute_monomial_polynomial_prod_torus(degree, p_tor, p_int, i);
                }
            }

            /// computes the coefficient of the monomial-th monomial of the product of p_tor and p_int (without FFT, quotient X^big_n + 1)
            fn compute_monomial_polynomial_prod_torus(
                degree: usize,
                p_tor: &[$T],
                p_int: &[$T],
                monomial: usize,
            ) -> $T {
                let mut res: $T = 0;
                for i in 0..degree {
                    // set the index of the X^j monomial from the i-th polynomial of the mask
                    let pow_tor: usize = i;
                    // set the index of the complementary monomial of the i-th polynomial of the secret key such that their product is a monomial of degree monomial
                    let mut pow_int: usize = monomial;
                    if monomial < i {
                        pow_int += degree; // mod big_n
                    }
                    pow_int -= i;

                    let tmp: $T = p_tor[pow_tor].wrapping_mul(p_int[pow_int]); //  torus::scalar_mul(&p_tor[pow_tor], );

                    if pow_int + pow_tor >= degree {
                        res = res.wrapping_mul(tmp); //  torus::sub_inplace(&mut res, &tmp);
                    } else {
                        res = res.wrapping_mul(tmp);
                        // torus::add_inplace(&mut res, &tmp);
                    }
                }
                return res;
            }

        }
    };
}

impl_trait_fft!(u32, "type Torus = u32;");
impl_trait_fft!(u64, "type Torus = u64;");

#[cfg(not(target_feature = "avx2"))]
/// Compute a mac (multiplier-accumulator): for all i, fft_res\[i\] += fft_a\[i\] * fft_b\[i\]
/// # Arguments
/// * `fft_res` - CTorus slice containing the result of the addition (output)
/// * `fft_a` - CTorus slice (input)
/// * `fft_b` - CTorus Aligned Vector (inputs)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_res: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// // ..
/// fft::mac(&mut fft_res, &fft_a, &fft_b);
/// ```
pub fn mac(
    fft_res: &mut AlignedVec<CTorus>,
    fft_a: &[CTorus], // &[Complex<f64>],
    fft_b: &AlignedVec<CTorus>,
) {
    for (coeff_tmp_ref, (coeff_a_i, coeff_b_i)) in
        fft_res.iter_mut().zip(fft_a.iter().zip(fft_b.iter()))
    {
        *coeff_tmp_ref += coeff_a_i * coeff_b_i;
    }
}

#[cfg(target_feature = "avx2")]
/// Compute a mac (multiplier-accumulator): for all i, fft_res\[i\] += fft_a\[i\] * fft_b\[i\]
/// # Arguments
/// * `fft_res` - CTorus slice containing the result of the addition (output)
/// * `fft_a` - CTorus slice (input)
/// * `fft_b` - CTorus Aligned Vector (inputs)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_res: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// // ..
/// fft::mac(&mut fft_res, &fft_a, &fft_b);
/// ```
pub fn mac(
    fft_res: &mut AlignedVec<CTorus>,
    fft_a: &[CTorus], // &[Complex<f64>],
    fft_b: &AlignedVec<CTorus>,
) {
    let n = fft_a.len();
    let index = n / 2 + 2;
    for (coeff_tmp_ref, (coeff_a_i, coeff_b_i)) in fft_res[..index]
        .chunks_mut(2)
        .zip(fft_a[..index].chunks(2).zip(fft_b[..index].chunks(2)))
    {
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

#[cfg(not(target_feature = "avx2"))]
#[inline]
/// Compute two mac (multiplier-accumulator)
/// fft_res_1\[i\] += fft_a_1\[i\] * fft_b\[i\] + fft_c_1\[i\] * fft_d\[i\]
/// fft_res_2\[i\] += fft_a_2\[i\] * fft_b\[i\] + fft_c_2\[i\] * fft_d\[i\]
/// # Arguments
/// * `fft_res_1`- Aligned Vector (output)
/// * `fft_res_2`- Aligned Vector (output)
/// * `fft_a_1` - CTorus slice (input)
/// * `fft_b`- Aligned Vector (input)
/// * `fft_c_1`- CTorus Slice (input)
/// * `fft_d`- Aligned Vector (input)
/// * `fft_a_2`- CTorus Slice (input)
/// * `fft_c_2`-  CTorus Slice (input)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_a_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_d: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_c_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_c_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_res_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_res_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// // ..
/// fft::two_double_mac(
///     &mut fft_res_1,
///     &mut fft_res_2,
///     &fft_a_1,
///     &fft_b,
///     &fft_c_1,
///     &fft_d,
///     &fft_a_2,
///     &fft_c_2,
/// );
/// ```
pub fn two_double_mac(
    fft_res_1: &mut AlignedVec<CTorus>,
    fft_res_2: &mut AlignedVec<CTorus>,
    fft_a_1: &[CTorus],
    fft_b: &AlignedVec<CTorus>,
    fft_c_1: &[CTorus],
    fft_d: &AlignedVec<CTorus>,
    fft_a_2: &[CTorus],
    fft_c_2: &[CTorus],
) {
    let n = fft_a_1.len();
    let index = n / 2 + 1;
    for (res_1_i, a_1_i, b_i, c_1_i, d_i, res_2_i, a_2_i, c_2_i) in izip!(
        fft_res_1[..index].iter_mut(),
        fft_a_1[..index].iter(),
        fft_b[..index].iter(),
        fft_c_1[..index].iter(),
        fft_d[..index].iter(),
        fft_res_2[..index].iter_mut(),
        fft_a_2[..index].iter(),
        fft_c_2[..index].iter(),
    ) {
        *res_1_i += a_1_i * b_i + c_1_i * d_i;
        *res_2_i += a_2_i * b_i + c_2_i * d_i;
    }
}

#[cfg(target_feature = "avx2")]
#[inline]
/// Compute two mac (multiplier-accumulator)
/// fft_res_1\[i\] += fft_a_1\[i\] * fft_b\[i\] + fft_c_1\[i\] * fft_d\[i\]
/// fft_res_2\[i\] += fft_a_2\[i\] * fft_b\[i\] + fft_c_2\[i\] * fft_d\[i\]
/// # Arguments
/// * `fft_res_1`- Aligned Vector (output)
/// * `fft_res_2`- Aligned Vector (output)
/// * `fft_a_1` - CTorus slice (input)
/// * `fft_b`- Aligned Vector (input)
/// * `fft_c_1`- CTorus Slice (input)
/// * `fft_d`- Aligned Vector (input)
/// * `fft_a_2`- CTorus Slice (input)
/// * `fft_c_2`-  CTorus Slice (input)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_a_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_d: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_c_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_c_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_res_1: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_res_2: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// // ..
/// fft::two_double_mac(
///     &mut fft_res_1,
///     &mut fft_res_2,
///     &fft_a_1,
///     &fft_b,
///     &fft_c_1,
///     &fft_d,
///     &fft_a_2,
///     &fft_c_2,
/// );
/// ```
pub fn two_double_mac(
    fft_res_1: &mut AlignedVec<CTorus>,
    fft_res_2: &mut AlignedVec<CTorus>,
    fft_a_1: &[CTorus],
    fft_b: &AlignedVec<CTorus>,
    fft_c_1: &[CTorus],
    fft_d: &AlignedVec<CTorus>,
    fft_a_2: &[CTorus],
    fft_c_2: &[CTorus],
) {
    let n = fft_a_1.len();
    let index = n / 2 + 2;
    // we take complex two by two
    for (res_1_i, a_1_i, b_i, c_1_i, d_i, res_2_i, a_2_i, c_2_i) in izip!(
        fft_res_1[..index].chunks_mut(2),
        fft_a_1[..index].chunks(2),
        fft_b[..index].chunks(2),
        fft_c_1[..index].chunks(2),
        fft_d[..index].chunks(2),
        fft_res_2[..index].chunks_mut(2),
        fft_a_2[..index].chunks(2),
        fft_c_2[..index].chunks(2),
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

            // Permute the real and imaginary elements of rlwe
            vec_b = _mm256_permute_pd(vec_b, 0x5);
            vec_d = _mm256_permute_pd(vec_d, 0x5);

            // Negate the imaginary elements of rlwe
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

#[cfg(not(target_feature = "avx2"))]
/// Compute fft_res +=  fft_a * fft_b + fft_c * fft_d
/// # Arguments
/// * `fft_res`- Aligned Vector (output)
/// * `fft_a` - CTorus slice (input)
/// * `fft_b`- Aligned Vector (input)
/// * `fft_c`- CTorus Slice (input)
/// * `fft_d`- Aligned Vector (input)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_d: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_c: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_res: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// // ..
/// fft::double_mac(&mut fft_res, &fft_a, &fft_b, &fft_c, &fft_d);
/// ```
pub fn double_mac(
    fft_res: &mut AlignedVec<CTorus>,
    fft_a: &[CTorus],
    fft_b: &AlignedVec<CTorus>,
    fft_c: &[CTorus],
    fft_d: &AlignedVec<CTorus>,
) {
    for (res_ref, (a_i, (b_i, (c_i, d_i)))) in fft_res.iter_mut().zip(
        fft_a
            .iter()
            .zip(fft_b.iter().zip(fft_c.iter().zip(fft_d.iter()))),
    ) {
        *res_ref += a_i * b_i + c_i * d_i;
    }
}

#[cfg(target_feature = "avx2")]
/// Compute fft_res +=  fft_a * fft_b + fft_c * fft_d
/// # Arguments
/// * `fft_res`- Aligned Vector (output)
/// * `fft_a` - CTorus slice (input)
/// * `fft_b`- Aligned Vector (input)
/// * `fft_c`- CTorus Slice (input)
/// * `fft_d`- Aligned Vector (input)
/// # Example
/// ```rust
/// #[macro_use]
/// use concrete;
/// use concrete::core_api::math::fft;
/// use concrete::types::{C2CPlanTorus, CTorus};
/// use fftw::array::AlignedVec;
///
/// let big_n: usize = 1024;
/// let mut fft_a: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_b: AlignedVec<CTorus> = AlignedVec::new(big_n);
/// let mut fft_d: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_c: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// let mut fft_res: AlignedVec<CTorus> = AlignedVec::new(big_n);
///
/// // ..
/// fft::double_mac(&mut fft_res, &fft_a, &fft_b, &fft_c, &fft_d);
/// ```
pub fn double_mac(
    fft_res: &mut AlignedVec<CTorus>,
    fft_a: &[CTorus],
    fft_b: &AlignedVec<CTorus>,
    fft_c: &[CTorus],
    fft_d: &AlignedVec<CTorus>,
) {
    let n = fft_a.len();
    let index = n / 2 + 2;
    for (res_ref, (a_i, (b_i, (c_i, d_i)))) in fft_res[..index].chunks_mut(2).zip(
        fft_a[..index].chunks(2).zip(
            fft_b[..index]
                .chunks(2)
                .zip(fft_c[..index].chunks(2).zip(fft_d[..index].chunks(2))),
        ),
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
