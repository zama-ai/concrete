//! Crossed Type Tensor Operations with RGSW samples already in the Fourier domain
//! * Contains functions dealing with different kind of tensors. For instance a function dealing with both LWE and RLWE samples is in here.

#[cfg(test)]
mod tests;

use crate::operators::math::{fft, PolynomialTensor, Tensor, FFT};
use crate::types::{C2CPlanTorus, CTorus, FTorus};
use crate::Types;
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use itertools::enumerate;
use num_integer::Integer;
use num_traits::Zero;
// use std::debug_assert;

pub trait Cross: Sized {
    fn external_product_inplace(
        res: &mut [Self],
        trgsw: &[CTorus],
        trlwe: &mut [Self],
        polynomial_size: usize,
        dimension: usize,
        base_log: usize,
        level: usize,
        fft: &mut C2CPlanTorus,
        ifft: &mut C2CPlanTorus,
        dec_i_fft: &mut AlignedVec<CTorus>,
        tmp_dec_i_fft: &mut AlignedVec<CTorus>,
        res_fft: &mut [AlignedVec<CTorus>],
    );
    fn external_product_inplace_one_dimension(
        res: &mut [Self],
        trgsw: &[CTorus],
        trlwe: &mut [Self],
        polynomial_size: usize,
        base_log: usize,
        level: usize,
        fft: &mut C2CPlanTorus,
        ifft: &mut C2CPlanTorus,
        mask_dec_i_fft: &mut AlignedVec<CTorus>,
        body_dec_i_fft: &mut AlignedVec<CTorus>,
        mask_res_fft: &mut AlignedVec<CTorus>,
        body_res_fft: &mut AlignedVec<CTorus>,
    );
    fn cmux_inplace(
        ct_0: &mut [Self],
        ct_1: &mut [Self],
        trgsw: &[CTorus],
        polynomial_size: usize,
        dimension: usize,
        base_log: usize,
        level: usize,
        fft: &mut C2CPlanTorus,
        ifft: &mut C2CPlanTorus,
        dec_i_fft: &mut AlignedVec<CTorus>,
        tmp_dec_i_fft: &mut AlignedVec<CTorus>,
        res_fft: &mut [AlignedVec<CTorus>],
    );
    fn blind_rotate_inplace(
        res: &mut [Self],
        lwe: &[Self],
        trgsw: &[CTorus],
        polynomial_size: usize,
        dimension: usize,
        base_log: usize,
        level: usize,
        fft: &mut C2CPlanTorus,
        ifft: &mut C2CPlanTorus,
        dec_i_fft: &mut AlignedVec<CTorus>,
        tmp_dec_i_fft: &mut AlignedVec<CTorus>,
        res_fft: &mut [AlignedVec<CTorus>],
    );
    fn constant_sample_extract(
        lwe: &mut [Self],
        rlwe: &[Self],
        dimension: usize,
        polynomial_size: usize,
    );
    fn bootstrap(
        lwe_out: &mut [Self],
        lwe_in: &[Self],
        trgsw: &[CTorus],
        base_log: usize,
        level: usize,
        accumulator: &mut [Self],
        polynomial_size: usize,
        dimension: usize,
    );
}

macro_rules! impl_trait_cross {
    ($T:ty,$DOC:expr) => {
        impl Cross for $T {
            /// Compute the external product between a trlwe and a trgsw (already in the fourier domain) and output a trlwe
            /// # Arguments
            /// * `res` - RLWE ciphertext (output)
            /// * `trgsw` - TRGSW ciphertext in the Fourier domain
            /// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
            /// * `dimension` - size of the rlwe mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of elements for the Torus decomposition
            /// * `fft`- FFTW Plan for FFT with CTorus element
            /// * `ifft`- FFTW Plan for IFFT with CTorus element
            /// * `dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `tmp_dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `res_fft`- AlignedVec used to store the intermediate result during the external product
            /// # Example
            /// ```rust
            /// use fftw::array::AlignedVec;
            /// use fftw::plan::C2CPlan;
            /// use fftw::types::{Flag, Sign};
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{Cross, SecretKey, RGSW, RLWE};
            /// use concrete_lib::operators::math::Tensor;
            /// use concrete_lib::types::{C2CPlanTorus, CTorus};
            /// use concrete_lib::Types;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 512;
            /// let dimension: usize = 2;
            /// let n_slots: usize = 1;
            /// let level: usize = 6;
            /// let base_log: usize = 4;
            /// let std_dev_bsk = f64::powi(2., -25);
            /// let std_dev_rlwe = f64::powi(2., -20);
            ///
            /// // creation of a rlwe secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // creation of a lwe secret key
            /// let mut lwe_sk: Vec<Torus> = vec![1 << (<Torus as Types>::TORUS_BIT - 1)];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // create the polynomial to encrypt
            /// let mut messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
            /// Tensor::uniform_random_default(&mut messages);
            ///
            /// // allocate space for the decrypted polynomial
            /// let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
            ///
            /// // allocation for the bootstrapping key
            /// let mut trgsw: Vec<CTorus> =
            ///     vec![CTorus::zero(); n_slots * dimension * (dimension + 1) * polynomial_size * level + n_slots * polynomial_size * level * (dimension + 1)];
            /// RGSW::create_fourier_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_bsk,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// // allocate vectors for rlwe ciphertexts (inputs)
            /// let mut ciphertexts: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            ///
            /// // allocate vectors for rlwe ciphertexts (outputs)
            /// let mut res: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            ///
            /// // encrypt the polynomial
            /// RLWE::sk_encrypt(
            ///     &mut ciphertexts,
            ///     &rlwe_sk,
            ///     &messages,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_rlwe,
            /// );
            ///
            /// // unroll FFT Plan using FFTW
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
            ///         .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
            ///         .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
            ///
            /// // allocate vectors used as temporary variables inside the external product
            /// let mut dec_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut tmp_dec_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut res_fft: Vec<AlignedVec<CTorus>> =
            ///     vec![AlignedVec::new(polynomial_size); dimension + 1];
            ///
            /// // external product
            /// Cross::external_product_inplace(
            ///     &mut res,
            ///     &trgsw,
            ///     &mut ciphertexts,
            ///     polynomial_size,
            ///     dimension,
            ///     base_log,
            ///     level,
            ///     &mut fft,
            ///     &mut ifft,
            ///     &mut dec_fft,
            ///     &mut tmp_dec_fft,
            ///     &mut res_fft,
            /// );
            /// ```
             fn external_product_inplace(
                res: &mut [$T],
                trgsw: &[CTorus],
                trlwe: &mut [$T],
                polynomial_size: usize,
                dimension: usize,
                base_log: usize,
                level: usize,
                fft: &mut C2CPlanTorus,
                ifft: &mut C2CPlanTorus,
                dec_i_fft: &mut AlignedVec<CTorus>,
                tmp_dec_i_fft: &mut AlignedVec<CTorus>,
                res_fft: &mut [AlignedVec<CTorus>],
            ) {
                // find the right twiddle factors for the given polynomial size
                let twiddles = TWIDDLES_TORUS!(polynomial_size);
                let inverse_twiddles = INVERSE_TWIDDLES_TORUS!(polynomial_size);
                // allocate space for the carry for the signed decomposition
                let mut carry: Vec<$T> = vec![0; polynomial_size * (dimension + 1)];
                let mut sign_decomp_0: Vec<$T> = vec![0; polynomial_size];
                let mut sign_decomp_1: Vec<$T> = vec![0; polynomial_size];

                // round mask and body
                for value in trlwe.iter_mut() {
                    *value = Types::round_to_closest_multiple(*value, base_log, level);
                }
                let matrix_size = (dimension + 1) * (dimension + 1) * polynomial_size;
                for (j, rgsw_level) in enumerate(izip!(
                    trgsw.chunks(matrix_size).rev(),
                )) {
                    let dec_level = level - j - 1;
                    // we work as long as possible with two polynomials to use put_2_in_fft_domain
                    // we work with one polynomial at a time for the last part of the computation
                    // when the dimension is even.
                    let trlwe_chunks = trlwe.chunks_exact(2* polynomial_size) ;
                    let carry_chunks_mut = carry.chunks_exact_mut(2 * polynomial_size) ;
                    let rgsw_level_chunks = rgsw_level.chunks_exact(2 * (polynomial_size * (dimension + 1))) ;

                    if dimension.is_even() {
                        // -----------> for the remainder
                        let rlwe_polynomial = trlwe_chunks.remainder() ;
                        let carry_polynomial = carry_chunks_mut.into_remainder();
                        let trgsw_line = rgsw_level_chunks.remainder() ;

                        // signed decomposition of a polynomial in the TRLWE mask
                        PolynomialTensor::signed_decompose_one_level(
                            &mut sign_decomp_0,
                            carry_polynomial,
                            rlwe_polynomial,
                            base_log,
                            dec_level,
                        );

                        // put the decomposition into the fft
                        // tmp_dec_i_fft is used as a temporary variable
                        FFT::put_in_fft_domain_storus(
                            dec_i_fft,
                            tmp_dec_i_fft,
                            &mut sign_decomp_0,
                            twiddles,
                            fft,
                        );
                        // do the element wise multiplication between polynomials in the fourier domain
                        for (trgsw_elt, res_fft_polynomial) in trgsw_line
                            .chunks(polynomial_size)
                            .zip(res_fft.iter_mut())
                        {
                            fft::mac(res_fft_polynomial, trgsw_elt, dec_i_fft);
                        }
                    }

                    // -----------> the exact chunks
                    for (double_rlwe_polynomial, double_carry_polynomial, double_trgsw_line) in izip!(
                        trlwe_chunks,
                        // carry_chunks_mut,
                        carry.chunks_exact_mut(2 * polynomial_size),
                        rgsw_level_chunks,
                    ) {
                        let (rlwe_polynomial_0, rlwe_polynomial_1) = double_rlwe_polynomial.split_at(polynomial_size) ;
                        let (carry_polynomial_0, carry_polynomial_1) = double_carry_polynomial.split_at_mut(polynomial_size) ;
                        let (trgsw_line_0, trgsw_line_1) = double_trgsw_line.split_at(polynomial_size * (dimension + 1)) ;

                        // signed decomposition of a polynomial in the TRLWE mask
                        PolynomialTensor::signed_decompose_one_level(
                            &mut sign_decomp_0,
                            carry_polynomial_0,
                            rlwe_polynomial_0,
                            base_log,
                            dec_level,
                        );
                        PolynomialTensor::signed_decompose_one_level(
                            &mut sign_decomp_1,
                            carry_polynomial_1,
                            rlwe_polynomial_1,
                            base_log,
                            dec_level,
                        );

                        // put the decomposition into the fft
                        // tmp_dec_i_fft is used as a temporary variable
                        FFT::put_2_in_fft_domain_storus(
                            dec_i_fft,
                            tmp_dec_i_fft,
                            &mut sign_decomp_0,
                            &mut sign_decomp_1,
                            twiddles,
                            fft,
                        );
                        // do the element wise multiplication between polynomials in the fourier domain
                        for (trgsw_elt_0, trgsw_elt_1, res_fft_polynomial) in izip!(trgsw_line_0
                            .chunks(polynomial_size),
                            trgsw_line_1
                            .chunks(polynomial_size),
                            res_fft.iter_mut())
                        {
                            fft::double_mac(res_fft_polynomial, trgsw_elt_0, dec_i_fft, trgsw_elt_1, tmp_dec_i_fft) ;
                            // fft::two_double_mac(&mut res_fft_0[0], &mut res_fft_1[0], trgsw_elt_0_0, dec_i_fft, trgsw_elt_0_1, tmp_dec_i_fft, trgsw_elt_1_0, trgsw_elt_1_1) ;
                            // fft::mac(res_fft_polynomial, trgsw_elt, dec_i_fft);
                        }
                        // do the element wise multiplication between polynomials in the fourier domain
                        // for (double_trgsw_elt_0, double_trgsw_elt_1, double_res_fft_polynomial) in izip!(trgsw_line_0
                        //     .chunks(2 * polynomial_size),
                        //     trgsw_line_1
                        //     .chunks(2 * polynomial_size),
                        //     res_fft.chunks_mut(2))
                        // {
                        //     let (trgsw_elt_0_0, trgsw_elt_0_1) = double_trgsw_elt_0.split_at(polynomial_size) ;
                        //     let (trgsw_elt_1_0, trgsw_elt_1_1) = double_trgsw_elt_1.split_at(polynomial_size) ;
                        //     let (res_fft_0, res_fft_1) = double_res_fft_polynomial.split_at_mut(1) ;
                        //     println!("trgsw_elt_0_0, trgsw_elt_0_1  = {}, {}", trgsw_elt_0_0.len(), trgsw_elt_0_1.len()) ;
                        //     println!("trgsw_elt_1_0, trgsw_elt_1_1 = {} {}", trgsw_elt_1_0.len(), trgsw_elt_1_1.len()) ;
                        //     println!("res_fft_0, res_fft_1 = {}, {}", res_fft_0[0].len(), res_fft_1[0].len()) ;
                        //     fft::two_double_mac(&mut res_fft_0[0], &mut res_fft_1[0], trgsw_elt_0_0, dec_i_fft, trgsw_elt_0_1, tmp_dec_i_fft, trgsw_elt_1_0, trgsw_elt_1_1) ;
                        //     // fft::mac(res_fft_polynomial, trgsw_elt, dec_i_fft);
                        // }
                    }
                }
                // we now have the result of the external product in (res_fft), we
                // convert it back to coefficient domain
                // mask
                if dimension.is_even()
                {
                    let res_remainder =  res
                    .chunks_exact_mut(2 * polynomial_size).into_remainder() ;
                    let res_fft_remainder = res_fft.chunks_exact_mut(2).into_remainder() ;
                    FFT::put_in_coeff_domain(res_remainder,dec_i_fft, &mut res_fft_remainder[0], inverse_twiddles, ifft) ;
                }

                for (double_res_polynomial, double_res_fft_polynomial) in res
                    .chunks_exact_mut(2 * polynomial_size)
                    .zip(res_fft.chunks_exact_mut(2))
                {
                    let (res_fft_0, res_fft_1) = double_res_fft_polynomial.split_at_mut(1) ;
                    let (res_0, res_1) = double_res_polynomial.split_at_mut(polynomial_size) ;
                    FFT::put_2_in_coeff_domain(res_0, res_1, &mut res_fft_0[0], &mut res_fft_1[0], inverse_twiddles, ifft) ;
                    // FFT::put_2_in_coeff_domain(
                    //     res_polynomial,
                    //     tmp_dec_i_fft,
                    //     res_fft_polynomial,
                    //     inverse_twiddles,
                    //     ifft,
                    // );
                }
            }


            /// Compute the external product between a trlwe and a trgsw (in the Fourier domain) and output a trlwe
            /// # Arguments
            /// * `res` -  RLWE ciphertext (output)
            /// * `trgsw` - TRGSW ciphertext in the Fourier domain
            /// * `polynomial_size` - number of coefficients of the polynomial
            /// * `dimension` - size of the mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `fft`- FFTW Plan for FFT with CTorus element
            /// * `ifft`- FFTW Plan for IFFT with CTorus element
            /// * `mask_dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `body_dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `mask_res_fft`- AlignedVec used to store the intermediate result during the external product
            /// * `body_res_fft`- AlignedVec used to store the intermediate result during the external product
            /// # Example
            /// ```rust
            /// use fftw::array::AlignedVec;
            /// use fftw::plan::C2CPlan;
            /// use fftw::types::{Flag, Sign};
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{Cross, SecretKey, RGSW, RLWE};
            /// use concrete_lib::operators::math::Tensor;
            /// use concrete_lib::types::{C2CPlanTorus, CTorus};
            /// use concrete_lib::Types;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 512;
            /// let dimension: usize = 1;
            /// let n_slots: usize = 1;
            /// let level: usize = 6;
            /// let base_log: usize = 4;
            /// let std_dev_bsk = f64::powi(2., -25);
            /// let std_dev_rlwe = f64::powi(2., -20);
            ///
            /// // creation of a rlwe secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            ///
            /// // creation of a lwe secret key
            /// let mut lwe_sk: Vec<Torus> = vec![1 << (<Torus as Types>::TORUS_BIT - 1)];
            /// Tensor::uniform_random_default(&mut lwe_sk);
            ///
            /// // create the polynomial to encrypt
            /// let mut messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
            /// Tensor::uniform_random_default(&mut messages);
            ///
            /// // allocate space for the decrypted polynomial
            /// let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
            ///
            /// // allocation for the bootstrapping key
            /// let mut trgsw: Vec<CTorus> =
            ///     vec![CTorus::zero(); n_slots * dimension * (dimension + 1) * polynomial_size * level + n_slots * polynomial_size * level * (dimension + 1)];
            /// RGSW::create_fourier_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_bsk,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            /// // allocate vectors for rlwe ciphertexts (inputs)
            /// let mut ciphertexts: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            ///
            /// // allocate vectors for rlwe ciphertexts (outputs)
            /// let mut res: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            ///
            /// // encrypt the polynomial
            /// RLWE::sk_encrypt(
            ///     &mut ciphertexts,
            ///     &rlwe_sk,
            ///     &messages,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_rlwe,
            /// );
            ///
            /// // unroll FFT Plan using FFTW
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
            ///         .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
            ///         .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
            ///
            /// // allocate vectors used as temporary variables inside the external product
            /// let mut mask_dec_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut body_dec_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut mask_res_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut body_res_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            ///
            /// // external product
            /// Cross::external_product_inplace_one_dimension(
            ///     &mut res,
            ///     &trgsw,
            ///     &mut ciphertexts,
            ///     polynomial_size,
            ///     base_log,
            ///     level,
            ///     &mut fft,
            ///     &mut ifft,
            ///     &mut mask_dec_fft,
            ///     &mut body_dec_fft,
            ///     &mut mask_res_fft,
            ///     &mut body_res_fft,
            /// );
            /// ```
             fn external_product_inplace_one_dimension(
                res: &mut [$T],
                trgsw: &[CTorus],
                trlwe: &mut [$T],
                polynomial_size: usize,
                base_log: usize,
                level: usize,
                fft: &mut C2CPlanTorus,  // &mut Radix4<f64>,
                ifft: &mut C2CPlanTorus, // &mut Radix4<f64>,
                mask_dec_i_fft: &mut AlignedVec<CTorus>,
                body_dec_i_fft: &mut AlignedVec<CTorus>,
                mask_res_fft: &mut AlignedVec<CTorus>,
                body_res_fft: &mut AlignedVec<CTorus>,
            ) {
                // find the right twiddle factors for the given polynomial size
                let twiddles = TWIDDLES_TORUS!(polynomial_size);
                let inverse_twiddles = INVERSE_TWIDDLES_TORUS!(polynomial_size);

                // references for trgsw blocks
                let trgsw_block0 = &trgsw[.. 2*level*polynomial_size] ;
                let trgsw_block1 = &trgsw[2 * level*polynomial_size..] ;

                // allocate space for the carry for the signed decomposition
                let mut mask_sign_decomp: Vec<$T> = vec![0; polynomial_size];
                let mut body_sign_decomp: Vec<$T> = vec![0; polynomial_size];
                let mut mask_carry: Vec<$T> = vec![0; polynomial_size];
                let mut body_carry: Vec<$T> = vec![0; polynomial_size];

                // round RLWE mask and body
                for value in trlwe.iter_mut() {
                    *value = Types::round_to_closest_multiple(*value, base_log, level);
                }

                let (mask_trlwe, body_trlwe) = trlwe.split_at(polynomial_size) ;
                let (mask_res, body_res) = res.split_at_mut(polynomial_size) ;

                // for each level in reverse order
                for (
                    j,
                    (
                        trgsw_line_0,
                        trgsw_line_1,
                    ),
                ) in enumerate(izip!(
                    trgsw_block0.chunks(2* polynomial_size).rev(),
                    trgsw_block1.chunks(2* polynomial_size).rev(),
                )) {
                    let (trgsw_mask_0, trgsw_body_0) = trgsw_line_0.split_at(polynomial_size) ;
                    let (trgsw_mask_1, trgsw_body_1) = trgsw_line_1.split_at(polynomial_size) ;

                    let dec_level = level - j - 1;
                    // sign decomposition of the trlwe mask
                    PolynomialTensor::signed_decompose_one_level(
                        &mut mask_sign_decomp,
                        &mut mask_carry,
                        mask_trlwe,
                        base_log,
                        dec_level,
                    );
                    // sign decomposition of the trlwe body
                    PolynomialTensor::signed_decompose_one_level(
                        &mut body_sign_decomp,
                        &mut body_carry,
                        body_trlwe,
                        base_log,
                        dec_level,
                    );

                    // put mask and body decompositions into the fourier domain
                    FFT::put_2_in_fft_domain_storus(
                        mask_dec_i_fft,
                        body_dec_i_fft,
                        &mask_sign_decomp,
                        &body_sign_decomp,
                        twiddles,
                        fft,
                    );
                    // perform element wise multiply between polynomials already in the fourier domain
                    // add the result into mask_res_fft and body_res_fft
                    fft::two_double_mac(
                        mask_res_fft,
                        body_res_fft,
                        trgsw_mask_0,
                        &mask_dec_i_fft,
                        trgsw_mask_1,
                        &body_dec_i_fft,
                        trgsw_body_0,
                        trgsw_body_1,
                    );
                }

                // we now have the result of the external product in (mask_res_fft, body_res_fft), we
                // convert it back to coefficient domain

                FFT::put_2_in_coeff_domain(
                    mask_res,
                    body_res,
                    mask_res_fft,
                    body_res_fft,
                    inverse_twiddles,
                    ifft,
                );
            }

            /// Compute the cmux gate between two RLWEs with a trgsw in the Fourier domain
            /// # Arguments
            /// * `ct_0` - first RLWE (output)
            /// * `ct_1` - second RLWE
            /// * `trgsw` - TRGSW ciphertext in the Fourier domain
            /// * `polynomial_size` - max degree of the polynomials + 1
            /// * `dimension` - size of the mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `fft`- FFTW Plan for FFT with CTorus element
            /// * `ifft`- FFTW Plan for IFFT with CTorus element
            /// * `dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `tmp_dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `res_fft`- AlignedVec used to store the intermediate result during the external product
            /// # Example
            /// ```rust
            /// use fftw::array::AlignedVec;
            /// use fftw::plan::C2CPlan;
            /// use fftw::types::{Flag, Sign};
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{Cross, SecretKey, RGSW, RLWE};
            /// use concrete_lib::operators::math::Tensor;
            /// use concrete_lib::types::{C2CPlanTorus, CTorus};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size = 512;
            /// let dimension: usize = 1;
            /// let n_slots = 1;
            /// let level = 4;
            /// let base_log = 7;
            /// let std_dev_bsk = f64::powi(2., -20);
            /// let std_dev_rlwe = f64::powi(2., -25);
            ///
            /// // compute the length of the rlwe secret key
            /// let rlwe_sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
            /// let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
            /// Tensor::uniform_random_default(&mut rlwe_sk);
            /// let lwe_sk = vec![0];
            ///
            /// // draw two random torus polynomials
            /// let mut m0: Vec<Torus> = vec![0; n_slots * polynomial_size];
            /// Tensor::uniform_random_default(&mut m0);
            /// let mut m1: Vec<Torus> = vec![0; n_slots * polynomial_size];
            /// Tensor::uniform_random_default(&mut m1);
            ///
            /// // allocation for the decrypted result
            /// let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
            ///
            /// // allocate and create the bootstrapping key
            /// let mut trgsw: Vec<CTorus> =
            ///     vec![CTorus::zero(); n_slots * dimension * (dimension + 1) * polynomial_size * level + n_slots * polynomial_size * level * (dimension + 1)];
            /// RGSW::create_fourier_bootstrapping_key(
            ///     &mut trgsw,
            ///     base_log,
            ///     level,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_bsk,
            ///     &lwe_sk,
            ///     &rlwe_sk,
            /// );
            ///
            /// // allocate rlwe vectors
            /// let mut ciphertexts0: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            /// let mut ciphertexts1: Vec<Torus> = vec![0; n_slots * (dimension + 1) * polynomial_size];
            ///
            /// // encrypt polynomials
            /// RLWE::sk_encrypt(
            ///     &mut ciphertexts0,
            ///     &rlwe_sk,
            ///     &m0,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_rlwe,
            /// );
            /// RLWE::sk_encrypt(
            ///     &mut ciphertexts1,
            ///     &rlwe_sk,
            ///     &m1,
            ///     dimension,
            ///     polynomial_size,
            ///     std_dev_rlwe,
            /// );
            ///
            /// // unroll FFT Plan using FFTW
            /// let mut fft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
            ///         .expect("test_cmux_0: C2CPlanTorus::aligned threw an error...");
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlanTorus::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
            ///         .expect("test_cmux_0: C2CPlanTorus::aligned threw an error...");
            ///
            /// // allocate vectors used as temporary variables inside the external product
            /// let mut dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut tmp_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut res_fft: Vec<AlignedVec<CTorus>> =
            ///     vec![AlignedVec::new(polynomial_size); dimension + 1];
            ///
            /// // compute cmux
            /// Cross::cmux_inplace(
            ///     &mut ciphertexts0,
            ///     &mut ciphertexts1,
            ///     &trgsw,
            ///     polynomial_size,
            ///     dimension,
            ///     base_log,
            ///     level,
            ///     &mut fft,
            ///     &mut ifft,
            ///     &mut dec_i_fft,
            ///     &mut tmp_dec_i_fft,
            ///     &mut res_fft,
            /// );
            /// ```
             fn cmux_inplace(
                ct_0: &mut [$T],
                ct_1: &mut [$T],
                trgsw: &[CTorus],
                polynomial_size: usize,
                dimension: usize,
                base_log: usize,
                level: usize,
                fft: &mut C2CPlanTorus,  // &mut Radix4<f64>,
                ifft: &mut C2CPlanTorus, // &mut Radix4<f64>,
                dec_i_fft: &mut AlignedVec<CTorus>,
                tmp_dec_i_fft: &mut AlignedVec<CTorus>,
                res_fft: &mut [AlignedVec<CTorus>],
            ) {
                // we perform C1 <- C1 - C0
                Tensor::sub_inplace(ct_1, ct_0);
                if dimension == 1 { // false { //
                    let (mask_res_fft, body_res_fft) = res_fft.split_at_mut(1) ;
                    // optimized external product for dimension = 1
                    Self::external_product_inplace_one_dimension(
                        ct_0,
                        trgsw,
                        ct_1,
                        polynomial_size,
                        base_log,
                        level,
                        fft,
                        ifft,
                        dec_i_fft,
                        tmp_dec_i_fft,
                        &mut mask_res_fft[0],
                        &mut body_res_fft[0],
                    );
                } else {
                    //generic external product working for all possible dimension
                    Self::external_product_inplace(
                        ct_0,
                        trgsw,
                        ct_1,
                        polynomial_size,
                        dimension,
                        base_log,
                        level,
                        fft,
                        ifft,
                        dec_i_fft,
                        tmp_dec_i_fft,
                        res_fft,
                    );
                }
            }

            /// Compute the blind rotate inplace with TRGSW samples in the Fourier domain
            /// * do not support batching for now
            /// * in this function we have ONLY ONE RLWE (body_res = 1 polynomial)
            /// # Arguments
            /// * `res` - TRLWE of the LUT, it will store the result of the blind rotate (output)
            /// * `lwe` - LWE containing the rotation
            /// * `trgsw` - TRGSWs of each bit of the secret key in the Fourier domain
            /// * `polynomial_size` - max degree of the polynomials + 1
            /// * `dimension` - size of the mask
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `fft`- FFTW Plan for FFT with CTorus element
            /// * `ifft`- FFTW Plan for IFFT with CTorus element
            /// * `dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `tmp_dec_i_fft`- temporary AlignedVec used to put polynomials into the Fourier domain
            /// * `res_fft`- AlignedVec used to store the intermediate result during the external product
            /// # Example
            /// ```rust
            /// use fftw::array::AlignedVec;
            /// use fftw::plan::C2CPlan;
            /// use fftw::types::{Flag, Sign};
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{cross, rgsw, rlwe, secret_key, Cross};
            /// use concrete_lib::operators::math::tensor;
            /// use concrete_lib::types::{C2CPlanTorus, CTorus};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size = 512;
            /// let dimension: usize = 1;
            /// let level: usize = 4;
            /// let base_log: usize = 7;
            /// let lwe_dimension: usize = 10;
            ///
            /// // creation of an accumulator
            /// let mut accumulator: Vec<Torus> = vec![0; polynomial_size * (dimension + 1)];
            ///
            /// // ... (fill the accumulator)
            ///
            /// // creation of an LWE sample
            /// let mut lwe_in: Vec<Torus> = vec![0; lwe_dimension + 1];
            ///
            /// // ... (fill this sample)
            ///
            /// // create a bootstrapping key
            /// let mut trgsw: Vec<CTorus> = vec![
            ///     CTorus::zero();
            ///     cross::get_bootstrapping_key_size(
            ///         dimension,
            ///         polynomial_size,
            ///         level,
            ///         lwe_dimension
            ///     )
            /// ];
            ///
            /// // ... (fill the key)
            ///
            /// // unroll fftw plan for the c2c FFT / IFFT
            /// let mut fft: C2CPlanTorus = C2CPlan::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
            ///     .expect("bootstrapp: C2CPlan::aligned threw an error...");
            /// let mut ifft: C2CPlanTorus =
            ///     C2CPlan::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
            ///         .expect("bootstrapp: C2CPlan::aligned threw an error...");
            ///
            /// // allocate temporary variable
            /// let mut dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut tmp_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
            /// let mut res_fft: Vec<AlignedVec<CTorus>> =
            ///     vec![AlignedVec::new(polynomial_size); dimension + 1];
            ///
            /// // compute blind rotate
            /// Cross::blind_rotate_inplace(
            ///     &mut accumulator,
            ///     &lwe_in,
            ///     &trgsw,
            ///     polynomial_size,
            ///     dimension,
            ///     base_log,
            ///     level,
            ///     &mut fft,
            ///     &mut ifft,
            ///     &mut dec_i_fft,
            ///     &mut tmp_dec_i_fft,
            ///     &mut res_fft,
            /// );
            /// ```
             fn blind_rotate_inplace(
                res: &mut [$T],
                lwe: &[$T],
                trgsw: &[CTorus],
                polynomial_size: usize,
                dimension: usize,
                base_log: usize,
                level: usize,
                fft: &mut C2CPlanTorus,
                ifft: &mut C2CPlanTorus,
                dec_i_fft: &mut AlignedVec<CTorus>,
                tmp_dec_i_fft: &mut AlignedVec<CTorus>,
                res_fft: &mut [AlignedVec<CTorus>],
            ) {
                let (&body_lwe, mask_lwe) = lwe.split_last().expect("Wrong length") ;

                // body_hat <- round(body * 2 * polynomial_size)
                let b_hat: usize = (body_lwe as FTorus / (<$T as Types>::TORUS_MAX as FTorus + 1.)
                    * 2.
                    * polynomial_size as FTorus)
                    .round() as usize;
                // compute ACC * X^(- body_hat)
                PolynomialTensor::divide_by_monomial(res, b_hat, polynomial_size);

                let mut ct_1 = vec![0; res.len()];

                let trgsw_size: usize = dimension * (dimension + 1) * level * polynomial_size + (dimension + 1) * level * polynomial_size;

                // for each trgsw i.e. for each bit of the lwe secret key
                for (a, trgsw_i) in mask_lwe.iter().zip(
                        trgsw
                        .chunks(trgsw_size)
                ) {
                    ct_1.copy_from_slice(res);
                    // a_hat <- round(a * 2 * polynomial_size)
                    let a_hat = (*a as FTorus / (<$T as Types>::TORUS_MAX as FTorus + 1.)
                        * 2.
                        * polynomial_size as FTorus)
                        .round() as usize;
                    if a_hat != 0 {
                        // compute ACC * X^{a_hat}
                        PolynomialTensor::multiply_by_monomial(&mut ct_1, a_hat, polynomial_size);
                        // we put 0. everywhere in mask_res_fft body_res_fft
                        for res_fft_polynomial in res_fft.iter_mut() {
                            for m in res_fft_polynomial
                                .iter_mut()
                            {
                                *m = CTorus::zero();
                            }
                        }
                        // select ACC or ACC * X^{a_hat} depending on the lwe secret key bit s
                        // i.e. return ACC * X^{a_hat * s}
                        Self::cmux_inplace(
                            res,
                            &mut ct_1,
                            trgsw_i,
                            polynomial_size,
                            dimension,
                            base_log,
                            level,
                            fft,
                            ifft,
                            dec_i_fft,
                            tmp_dec_i_fft,
                            res_fft,
                        );
                    }
                }
            }

            /// Compute the sample extraction of the constant monomial
            /// # Arguments
            /// * `lwe` - output LWE (output)
            /// * `rlwe` - input RLWE
            /// * `dimension` - size of the rlwe mask
            /// * `polynomial_size` - max degree of the polynomials + 1
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::crypto::{Cross, rlwe, secret_key};
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size: usize = 512;
            /// let dimension = 2 ;
            /// // create a rlwe sample
            /// let mut rlwe: Vec<Torus> = vec![0; polynomial_size * (dimension + 1)];
            ///
            /// // ... (fill the rlwe sample)
            ///
            /// // create a lwe sample to store the output
            /// let mut lwe_out: Vec<Torus> = vec![0; polynomial_size * dimension + 1];
            ///
            /// // extract the constant monomial
            /// Cross::constant_sample_extract(
            ///     &mut lwe_out,
            ///     &rlwe,
            ///     dimension,
            ///     polynomial_size,
            /// );
            /// ```
             fn constant_sample_extract(
                lwe: &mut [$T],
                rlwe: &[$T],
                dimension: usize,
                polynomial_size: usize,
            ) {
                let (body_lwe, mask_lwe) = lwe.split_last_mut().expect("Wrong length") ;
                let (mask_rlwe, body_rlwe) = rlwe.split_at(polynomial_size * dimension) ;

                for (mask_lwe_polynomial, mask_rlwe_polynomial) in mask_lwe
                    .chunks_mut(polynomial_size)
                    .zip(mask_rlwe.chunks(polynomial_size))
                {
                    for (lwe_coeff, rlwe_coeff) in mask_lwe_polynomial
                        .iter_mut()
                        .zip(mask_rlwe_polynomial.iter().rev())
                    {
                        *lwe_coeff = (0 as $T).wrapping_sub(*rlwe_coeff);
                    }
                }

                PolynomialTensor::multiply_by_monomial(mask_lwe, 1, polynomial_size);
                *body_lwe = body_rlwe[0];
            }

            /// Compute the bootstrapping on a lwe sample allowing to reduce the noise
            /// and apply arbitrary function on it
            /// # Arguments
            /// * `lwe_body_out` - output LWE (output)
            /// * `lwe_body_in` - LWE to bootstrap
            /// * `trgsw` - TRGSWs of each bit of the secret key, the mask of the BSK in the Fourier domain
            /// * `base_log` - decomposition base of the gadget matrix
            /// * `level` - number of blocks of the gadget matrix
            /// * `accumulator` - accumulator
            /// * `polynomial_size` - max degree of the polynomials + 1
            /// * `dimension` - size of the mask
            /// Example
            /// ```rust
            /// use num_traits::Zero;
            /// use concrete_lib::operators::crypto::{cross, rgsw, rlwe, secret_key, Cross};
            /// use concrete_lib::types::CTorus;
            #[doc = $DOC]
            ///
            /// // settings
            /// let polynomial_size = 512;
            /// let dimension: usize = 1;
            /// let level: usize = 4;
            /// let base_log: usize = 7;
            /// let lwe_dimension: usize = 10;
            ///
            /// // creation of a LWE sample for the input
            /// let mut lwe_in: Vec<Torus> = vec![0; lwe_dimension + 1];
            ///
            /// // ... (fill the sample)
            ///
            /// // creation of an accumulator
            /// let mut accumulator: Vec<Torus> = vec![0; polynomial_size * (dimension + 1)];
            ///
            /// // ... (fill the accumulator)
            ///
            /// // create a bootstrapping key
            /// let mut trgsw: Vec<CTorus> = vec![
            ///     CTorus::zero();
            ///     cross::get_bootstrapping_key_size(
            ///         dimension,
            ///         polynomial_size,
            ///         level,
            ///         lwe_dimension
            ///     )
            /// ];
            ///
            /// // ... (fill the key)
            ///
            /// // creation of a LWE sample for the output
            /// let mut lwe_out: Vec<Torus> = vec![0; dimension * polynomial_size + 1];
            ///
            /// // execute the bootstrap
            /// Cross::bootstrap(
            ///     &mut lwe_out,
            ///     &lwe_in,
            ///     &trgsw,
            ///     base_log,
            ///     level,
            ///     &mut accumulator,
            ///     polynomial_size,
            ///     dimension,
            /// );
            /// ```
             fn bootstrap(
                lwe_out: &mut [$T],
                lwe_in: &[$T],
                trgsw: &[CTorus],
                base_log: usize,
                level: usize,
                accumulator: &mut [$T],
                polynomial_size: usize,
                dimension: usize,
            ) {
                // unroll fftw plan for the c2c FFT / IFFT
                let mut fft: C2CPlanTorus = C2CPlan::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
                    .expect("bootstrapp: C2CPlan::aligned threw an error...");
                let mut ifft: C2CPlanTorus =
                    C2CPlan::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
                        .expect("bootstrapp: C2CPlan::aligned threw an error...");
                // allocate temporary variable
                let mut dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                let mut tmp_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                let mut res_fft: Vec<AlignedVec<CTorus>> =
                    vec![AlignedVec::new(polynomial_size); dimension + 1];
                // compute blind rotate
                Self::blind_rotate_inplace(
                    accumulator,
                    lwe_in,
                    trgsw,
                    polynomial_size,
                    dimension,
                    base_log,
                    level,
                    &mut fft,
                    &mut ifft,
                    &mut dec_i_fft,
                    &mut tmp_dec_i_fft,
                    &mut res_fft,
                );
                // extract the constant monomial
                Self::constant_sample_extract(
                    lwe_out,
                    accumulator,
                    dimension,
                    polynomial_size,
                );
            }
        }
    };
}

impl_trait_cross!(u32, "type Torus = u32;");
impl_trait_cross!(u64, "type Torus = u64;");

#[no_mangle]
/// Computes the size of a bootstrapping key
/// # Arguments
/// * `dimension` - size of the mask
/// * `polynomial_size` - number of coefficients of the polynomial
/// * `level` - number of blocks of the gadget matrix
/// * `lwe_sk_n_bits` - size of the LWE secret key
/// # Output
/// * the size
/// # Example
/// ```rust
/// use concrete_lib::operators::crypto::cross;
///
/// // settings
/// let polynomial_size = 512;
/// let dimension: usize = 1;
/// let level: usize = 4;
/// let lwe_dimension: usize = 10;
///
/// let size = cross::get_bootstrapping_key_size(dimension, polynomial_size, level, lwe_dimension);
/// ```
pub fn get_bootstrapping_key_size(
    dimension: usize,
    polynomial_size: usize,
    level: usize,
    lwe_sk_n_bits: usize,
) -> usize {
    return dimension * (dimension + 1) * polynomial_size * level * lwe_sk_n_bits
        + polynomial_size * level * (dimension + 1) * lwe_sk_n_bits;
}
