//! Operations involving ciphertexts of different schemes.

use concrete_commons::{CastInto, Numeric};

use crate::crypto::UnsignedTorus;
use crate::math::decomposition::{
    DecompositionBaseLog, DecompositionLevel, DecompositionLevelCount,
};
use crate::math::fft::{Complex64, Fft, FourierPolynomial};
use crate::math::polynomial::{MonomialDegree, Polynomial, PolynomialList};
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor};
use crate::{ck_dim_eq, zip, zip_args};

use super::bootstrap::BootstrapKey;
use super::ggsw::GgswCiphertext;
use super::glwe::GlweCiphertext;
use super::lwe::{LweBody, LweCiphertext};

#[cfg(test)]
mod tests;

/// Executes the external product of a GLWE ciphertext with a GGSW ciphertext.
pub fn external_product<RgswCont, RlweCont, InCont, FftCont1, FftCont2, FftCont3, Scalar>(
    fft: &mut Fft,
    dec_i_fft: &mut FourierPolynomial<FftCont1>,
    tmp_dec_i_fft: &mut FourierPolynomial<FftCont2>,
    res_fft: &mut [FourierPolynomial<FftCont3>],
    output: &mut GlweCiphertext<InCont>,
    ggsw: &GgswCiphertext<RgswCont>,
    glwe: &mut GlweCiphertext<RlweCont>,
) where
    GlweCiphertext<InCont>: AsMutTensor<Element = Scalar>,
    GgswCiphertext<RgswCont>: AsRefTensor<Element = Complex64>,
    GlweCiphertext<RlweCont>: AsMutTensor<Element = Scalar>,
    FourierPolynomial<FftCont1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont2>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont3>: AsMutTensor<Element = Complex64>,
    Scalar: UnsignedTorus,
{
    ck_dim_eq!(glwe.polynomial_size().0 => ggsw.polynomial_size().0);
    ck_dim_eq!(output.polynomial_size().0 => ggsw.polynomial_size().0);
    ck_dim_eq!(glwe.size().0 => ggsw.glwe_size().0);
    ck_dim_eq!(output.size().0 => ggsw.glwe_size().0);

    // We retrieve the parameters from the RGSW.
    let base_log = ggsw.decomposition_base_log().0;
    let level = ggsw.decomposition_level_count().0;
    let polynomial_size = glwe.polynomial_size().0;
    let dimension = glwe.mask_size().0;
    let even_dimension = dimension % 2 == 0;

    // allocate space for the carry for the signed decomposition
    let zero = <Scalar as Numeric>::ZERO;
    let mut carry = vec![zero; polynomial_size * (dimension + 1)];
    let mut sign_decomp_0 = vec![zero; polynomial_size];
    let mut sign_decomp_1 = vec![zero; polynomial_size];

    // round mask and body
    for value in glwe.as_mut_tensor().as_mut_slice().iter_mut() {
        *value = value.round_to_closest_multiple(
            DecompositionBaseLog(base_log),
            DecompositionLevelCount(level),
        );
    }

    let matrix_size = (dimension + 1) * (dimension + 1) * polynomial_size;

    for (j, rgsw_level) in ggsw
        .as_tensor()
        .as_slice()
        .chunks(matrix_size)
        .rev()
        .enumerate()
    {
        let dec_level = level - j - 1;
        // we work as long as possible with two polynomials to use put_2_in_fft_domain
        // we work with one polynomial at a time for the last part of the computation
        // when the dimension is even.
        let trlwe_chunks = glwe
            .as_tensor()
            .as_slice()
            .chunks_exact(2 * polynomial_size);
        let carry_chunks_mut = carry.chunks_exact_mut(2 * polynomial_size);
        let rgsw_level_chunks = rgsw_level.chunks_exact(2 * (polynomial_size * (dimension + 1)));

        if even_dimension {
            // -----------> for the remainder
            let rlwe_polynomial = trlwe_chunks.remainder();
            let carry_polynomial = carry_chunks_mut.into_remainder();
            let trgsw_line = rgsw_level_chunks.remainder();

            // signed decomposition of a polynomial in the TRLWE mask
            signed_decompose_one_level(
                &mut sign_decomp_0,
                carry_polynomial,
                rlwe_polynomial,
                DecompositionBaseLog(base_log),
                DecompositionLevel(dec_level),
            );

            // put the decomposition into the fft
            // tmp_dec_i_fft is used as a temporary variable
            fft.forward_as_integer(
                dec_i_fft,
                &Polynomial::from_container(sign_decomp_0.as_slice()),
            );
            // do the element wise multiplication between polynomials in the fourier domain
            for (trgsw_elt, res_fft_polynomial) in
                trgsw_line.chunks(polynomial_size).zip(res_fft.iter_mut())
            {
                res_fft_polynomial.update_with_multiply_accumulate(
                    &FourierPolynomial::from_container(trgsw_elt),
                    dec_i_fft,
                );
            }
        }

        // -----------> the exact chunks
        for zip_args!(
            double_rlwe_polynomial,
            double_carry_polynomial,
            double_trgsw_line
        ) in zip!(
            trlwe_chunks,
            carry.chunks_exact_mut(2 * polynomial_size),
            rgsw_level_chunks
        ) {
            let (rlwe_polynomial_0, rlwe_polynomial_1) =
                double_rlwe_polynomial.split_at(polynomial_size);
            let (carry_polynomial_0, carry_polynomial_1) =
                double_carry_polynomial.split_at_mut(polynomial_size);
            let (trgsw_line_0, trgsw_line_1) =
                double_trgsw_line.split_at(polynomial_size * (dimension + 1));

            // signed decomposition of a polynomial in the TRLWE mask
            signed_decompose_one_level(
                &mut sign_decomp_0,
                carry_polynomial_0,
                rlwe_polynomial_0,
                DecompositionBaseLog(base_log),
                DecompositionLevel(dec_level),
            );
            signed_decompose_one_level(
                &mut sign_decomp_1,
                carry_polynomial_1,
                rlwe_polynomial_1,
                DecompositionBaseLog(base_log),
                DecompositionLevel(dec_level),
            );

            // put the decomposition into the fft
            // tmp_dec_i_fft is used as a temporary variable
            fft.forward_two_as_integer(
                dec_i_fft,
                tmp_dec_i_fft,
                &Polynomial::from_container(sign_decomp_0.as_slice()),
                &Polynomial::from_container(sign_decomp_1.as_slice()),
            );
            // do the element wise multiplication between polynomials in the fourier domain
            for zip_args!(trgsw_elt_0, trgsw_elt_1, res_fft_polynomial) in zip!(
                trgsw_line_0.chunks(polynomial_size),
                trgsw_line_1.chunks(polynomial_size),
                res_fft.iter_mut()
            ) {
                res_fft_polynomial.update_with_two_multiply_accumulate(
                    &FourierPolynomial::from_container(trgsw_elt_0),
                    dec_i_fft,
                    &FourierPolynomial::from_container(trgsw_elt_1),
                    tmp_dec_i_fft,
                );
            }
        }
    }
    // we now have the result of the external product in (res_fft), we
    // convert it back to coefficient domain
    // mask
    if even_dimension {
        let res_remainder = output
            .as_mut_tensor()
            .as_mut_slice()
            .chunks_exact_mut(2 * polynomial_size)
            .into_remainder();
        let res_fft_remainder = res_fft.chunks_exact_mut(2).into_remainder();
        fft.add_backward_as_torus(
            &mut Polynomial::from_container(res_remainder),
            &mut res_fft_remainder[0],
        );
    }
    for (double_res_polynomial, double_res_fft_polynomial) in output
        .as_mut_tensor()
        .as_mut_slice()
        .chunks_exact_mut(2 * polynomial_size)
        .zip(res_fft.chunks_exact_mut(2))
    {
        let (res_fft_0, res_fft_1) = double_res_fft_polynomial.split_at_mut(1);
        let (res_0, res_1) = double_res_polynomial.split_at_mut(polynomial_size);
        let mut res_0 = Polynomial::from_container(res_0);
        let mut res_1 = Polynomial::from_container(res_1);
        fft.add_backward_two_as_torus(&mut res_0, &mut res_1, &mut res_fft_0[0], &mut res_fft_1[0]);
    }
}

/// Executes the CMUX operations of two GLWE ciphertexts conditioned on a GGSW ciphertext
///
/// # Note
///
/// The result is stored in the `glwe_0` ciphertext.
pub fn cmux<RlweCont0, RlweCont1, RgswCont, FftCont1, FftCont2, FftCont3, Scalar>(
    fft: &mut Fft,
    dec_i_fft: &mut FourierPolynomial<FftCont1>,
    tmp_dec_i_fft: &mut FourierPolynomial<FftCont2>,
    res_fft: &mut [FourierPolynomial<FftCont3>],
    glwe_0: &mut GlweCiphertext<RlweCont0>,
    glwe_1: &mut GlweCiphertext<RlweCont1>,
    ggsw: &GgswCiphertext<RgswCont>,
) where
    GgswCiphertext<RgswCont>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<FftCont1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont2>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont3>: AsMutTensor<Element = Complex64>,
    GlweCiphertext<RlweCont0>: AsMutTensor<Element = Scalar>,
    GlweCiphertext<RlweCont1>: AsMutTensor<Element = Scalar>,
    Scalar: UnsignedTorus,
{
    // we perform C1 <- C1 - C0
    glwe_1
        .as_mut_tensor()
        .update_with_wrapping_sub(glwe_0.as_tensor());
    //generic external product working for all possible dimension
    external_product(fft, dec_i_fft, tmp_dec_i_fft, res_fft, glwe_0, ggsw, glwe_1);
}

/// Fills the `output` ciphertext with the result of the blind rotation of the bootstrap key by
/// the LWE ciphertext.
pub fn blind_rotate<OutCont, LweCont, BskCont, FftCont1, FftCont2, FftCont3, Scalar>(
    fft: &mut Fft,
    dec_i_fft: &mut FourierPolynomial<FftCont1>,
    tmp_dec_i_fft: &mut FourierPolynomial<FftCont2>,
    res_fft: &mut [FourierPolynomial<FftCont3>],
    output: &mut GlweCiphertext<OutCont>,
    lwe: &LweCiphertext<LweCont>,
    bootstrap_key: &BootstrapKey<BskCont>,
) where
    GlweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
    GlweCiphertext<Vec<Scalar>>: AsMutTensor<Element = Scalar>,
    LweCiphertext<LweCont>: AsRefTensor<Element = Scalar>,
    BootstrapKey<BskCont>: AsRefTensor<Element = Complex64>,
    FourierPolynomial<FftCont1>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont2>: AsMutTensor<Element = Complex64>,
    FourierPolynomial<FftCont3>: AsMutTensor<Element = Complex64>,
    Scalar: UnsignedTorus,
{
    // We retrieve dimensions
    let dimension = output.mask_size().0;
    let level = bootstrap_key.level_count().0;
    let polynomial_size = output.polynomial_size().0;

    let (body_lwe, mask_lwe) = lwe.get_body_and_mask();

    // body_hat <- round(body * 2 * polynomial_size)
    let n_coefs: f64 = output.polynomial_size().0.cast_into();
    let tmp: f64 = body_lwe.0.cast_into() / (<Scalar as Numeric>::MAX.cast_into() + 1.);
    let tmp: f64 = tmp * 2. * n_coefs;
    let b_hat: usize = tmp.round().cast_into();

    // compute ACC * X^(- body_hat)
    output
        .as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(MonomialDegree(b_hat));

    let mut ct_1 = GlweCiphertext::allocate(Scalar::ZERO, output.polynomial_size(), output.size());

    let trgsw_size: usize = dimension * (dimension + 1) * level * polynomial_size
        + (dimension + 1) * level * polynomial_size;

    // for each trgsw i.e. for each bit of the lwe secret key
    for (a, trgsw_i) in mask_lwe
        .mask_element_iter()
        .zip(bootstrap_key.as_tensor().as_slice().chunks(trgsw_size))
    {
        ct_1.as_mut_tensor()
            .as_mut_slice()
            .copy_from_slice(output.as_tensor().as_slice());
        // a_hat <- round(a * 2 * polynomial_size)
        let poly_size: f64 = polynomial_size.cast_into();
        let tmp: f64 = (*a).cast_into() / (<Scalar as Numeric>::MAX.cast_into() + 1.);
        let tmp: f64 = tmp * 2. * poly_size;
        let a_hat: usize = tmp.round().cast_into();
        if a_hat != 0 {
            // compute ACC * X^{a_hat}
            ct_1.as_mut_polynomial_list()
                .update_with_wrapping_monic_monomial_mul(MonomialDegree(a_hat));
            // we put 0. everywhere in mask_res_fft body_res_fft
            for res_fft_polynomial in res_fft.iter_mut() {
                for m in res_fft_polynomial.coefficient_iter_mut() {
                    *m = Complex64::new(0., 0.);
                }
            }
            // select ACC or ACC * X^{a_hat} depending on the lwe secret key bit s
            // i.e. return ACC * X^{a_hat * s}
            cmux(
                fft,
                dec_i_fft,
                tmp_dec_i_fft,
                res_fft,
                output,
                &mut ct_1,
                &GgswCiphertext::from_container(
                    trgsw_i,
                    bootstrap_key.glwe_size(),
                    bootstrap_key.polynomial_size(),
                    bootstrap_key.base_log(),
                ),
            );
        }
    }
}

/// Extracts the constant term of a GLWE ciphertext into an LWE ciphertext.
pub fn constant_sample_extract<LweCont, RlweCont, Scalar>(
    lwe: &mut LweCiphertext<LweCont>,
    glwe: &GlweCiphertext<RlweCont>,
) where
    LweCiphertext<LweCont>: AsMutTensor<Element = Scalar>,
    GlweCiphertext<RlweCont>: AsRefTensor<Element = Scalar>,
    Scalar: UnsignedTorus,
{
    let (body_lwe, mut mask_lwe) = lwe.get_mut_body_and_mask();
    let (body_rlwe, mask_rlwe) = glwe.get_body_and_mask();
    let polynomial_size = glwe.polynomial_size().0;

    for (mask_lwe_polynomial, mask_rlwe_polynomial) in mask_lwe
        .as_mut_tensor()
        .as_mut_slice()
        .chunks_mut(polynomial_size)
        .zip(mask_rlwe.as_tensor().as_slice().chunks(polynomial_size))
    {
        for (lwe_coeff, rlwe_coeff) in mask_lwe_polynomial
            .iter_mut()
            .zip(mask_rlwe_polynomial.iter().rev())
        {
            *lwe_coeff = (Scalar::ZERO).wrapping_sub(*rlwe_coeff);
        }
    }

    let mut mask_lwe_poly = PolynomialList::from_container(
        mask_lwe.as_mut_tensor().as_mut_slice(),
        glwe.polynomial_size(),
    );
    mask_lwe_poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(1));
    *body_lwe = LweBody(*body_rlwe.as_tensor().get_element(0));
}

/// Performs the bootstrapping of an LWE ciphertext, with a bootstrapping key.
///
/// # Example
///
/// ```
/// use concrete_core::crypto::bootstrap::BootstrapKey;
/// use concrete_core::crypto::{GlweSize, LweSize, LweDimension, GlweDimension};
/// use concrete_core::math::decomposition::{DecompositionLevelCount, DecompositionBaseLog};
/// use concrete_core::math::polynomial::PolynomialSize;
/// use concrete_core::crypto::secret::{LweSecretKey, GlweSecretKey};
/// use concrete_core::crypto::lwe::LweCiphertext;
/// use concrete_core::crypto::glwe::GlweCiphertext;
/// use concrete_core::crypto::cross::bootstrap;
/// use concrete_core::math::random::{RandomGenerator, EncryptionRandomGenerator};
/// use concrete_core::math::fft::Complex64;
/// use concrete_commons::LogStandardDev;
///
/// let mut generator = RandomGenerator::new(None);
/// let mut secret_gen = EncryptionRandomGenerator::new(None);
///
/// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(512));
/// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
/// let mut bsk = BootstrapKey::allocate(
///     9u32,
///     glwe_dim.to_glwe_size(),
///     poly_size,
///     dec_lc,
///     dec_bl,
///     lwe_dim
/// );
/// let lwe_sk = LweSecretKey::generate(lwe_dim, &mut generator);
/// let glwe_sk = GlweSecretKey::generate(glwe_dim, poly_size, &mut generator);
/// bsk.fill_with_new_key(
///     &lwe_sk,
///     &glwe_sk,
///     LogStandardDev::from_log_standard_dev(-15.),
///     &mut secret_gen
/// );
/// let mut frr_bsk = BootstrapKey::allocate_complex(
///     Complex64::new(0.,0.),
///     glwe_dim.to_glwe_size(),
///     poly_size,
///     dec_lc,
///     dec_bl,
///     lwe_dim
/// );
/// frr_bsk.fill_with_forward_fourier(&bsk);
/// let lwe_in = LweCiphertext::allocate(9u32, lwe_dim.to_lwe_size());
/// let mut lwe_out = LweCiphertext::allocate(9u32, LweSize(glwe_dim.0 * poly_size.0 + 1));
/// let mut accumulator = GlweCiphertext::allocate(0u32, poly_size, glwe_dim.to_glwe_size());
/// bootstrap(&mut lwe_out, &lwe_in, &frr_bsk, &mut accumulator);
/// ```
pub fn bootstrap<OutCont, InCont, BskCont, AccCont, Scalar>(
    lwe_out: &mut LweCiphertext<OutCont>,
    lwe_in: &LweCiphertext<InCont>,
    bootstrap_key: &BootstrapKey<BskCont>,
    accumulator: &mut GlweCiphertext<AccCont>,
) where
    LweCiphertext<OutCont>: AsMutTensor<Element = Scalar>,
    LweCiphertext<InCont>: AsRefTensor<Element = Scalar>,
    BootstrapKey<BskCont>: AsRefTensor<Element = Complex64>,
    GlweCiphertext<AccCont>: AsMutTensor<Element = Scalar>,
    Scalar: UnsignedTorus,
{
    let polynomial_size = bootstrap_key.polynomial_size();
    let dimension = bootstrap_key.glwe_size().0 - 1;

    // unroll fftw plan for the c2c FFT / IFFT
    let mut fft = Fft::new(polynomial_size);

    // allocate temporary variable
    let mut dec_i_fft = FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size);
    let mut tmp_dec_i_fft = FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size);
    let mut res_fft =
        vec![FourierPolynomial::allocate(Complex64::new(0., 0.), polynomial_size); dimension + 1];

    // compute blind rotate
    blind_rotate(
        &mut fft,
        &mut dec_i_fft,
        &mut tmp_dec_i_fft,
        &mut res_fft,
        accumulator,
        lwe_in,
        bootstrap_key,
    );

    // extract the constant monomial
    constant_sample_extract(lwe_out, accumulator);
}

fn signed_decompose_one_level<Scalar>(
    sign_decomp: &mut [Scalar],
    carries: &mut [Scalar],
    polynomial: &[Scalar],
    base_log: DecompositionBaseLog,
    dec_level: DecompositionLevel,
) where
    Scalar: UnsignedTorus,
{
    // loop over the coefficients of the polynomial
    for (carry, (decomp, value)) in carries
        .iter_mut()
        .zip(sign_decomp.iter_mut().zip(polynomial.iter()))
    {
        let pair = value.signed_decompose_one_level(*carry, base_log, dec_level);
        *decomp = pair.0;
        *carry = pair.1;
    }
}
