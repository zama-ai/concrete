/// Contains material needed to estimate the growth of the noise when performing homomorphic
/// computation
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::{CastInto, UnsignedInteger};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};

use super::*;

/// Computes the dispersion of an addition of two
/// uncorrelated ciphertexts.
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_npe::estimate_addition_noise;
/// let var1 = Variance(2_f64.powf(-25.));
/// let var2 = Variance(2_f64.powf(-25.));
/// let var_out = estimate_addition_noise::<u64, _, _>(var1, var2);
/// println!("Expect Variance (2^24) =  {}", f64::powi(2., -24));
/// println!("Output Variance {}", var_out.get_variance());
/// assert!((f64::powi(2., -24) - var_out.get_variance()).abs() < 0.0001);
/// ```
pub fn estimate_addition_noise<T, D1, D2>(dispersion_ct1: D1, dispersion_ct2: D2) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    // The result variance is equal to the sum of the input variances
    let var_res: f64 =
        dispersion_ct1.get_modular_variance::<T>() + dispersion_ct2.get_modular_variance::<T>();
    Variance::from_modular_variance::<T>(var_res)
}

/// Computes the dispersion of an addition of
/// several uncorrelated ciphertexts.
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_npe::estimate_several_additions_noise;
/// let var1 = Variance(2_f64.powf(-25.));
/// let var2 = Variance(2_f64.powf(-25.));
/// let var3 = Variance(2_f64.powf(-24.));
/// let var_in = [var1, var2, var3];
/// let var_out = estimate_several_additions_noise::<u64, _>(&var_in);
/// println!("Expect Variance (2^24) =  {}", f64::powi(2., -23));
/// println!("Output Variance {}", var_out.get_variance());
/// assert!((f64::powi(2., -23) - var_out.get_variance()).abs() < 0.0001);
/// ```
pub fn estimate_several_additions_noise<T, D>(dispersion_cts: &[D]) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;
    // The result variance is equal to the sum of the input variances
    for dispersion in dispersion_cts.iter() {
        var_res += dispersion.get_modular_variance::<T>();
    }
    Variance::from_modular_variance::<T>(var_res)
}

/// Computes the dispersion of a multiplication
/// of a ciphertext by a scalar.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_npe::estimate_integer_plaintext_multiplication_noise;
/// let variance = Variance(f64::powi(2., -48));
/// let n: u64 = 543;
/// // noise computation
/// let var_out = estimate_integer_plaintext_multiplication_noise::<u64, _>(variance, n);
/// ```
pub fn estimate_integer_plaintext_multiplication_noise<T, D>(variance: D, n: T) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let sn = n.into_signed();
    let product: f64 = (sn * sn).cast_into();
    Variance::from_variance(variance.get_variance() * product)
}

/// Computes the dispersion of a multisum between
/// uncorrelated ciphertexts and scalar weights $w_i$ i.e.,  $\sigma_{out}^2 = \sum_i w_i^2 *
/// \sigma_i^2$.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_npe::estimate_weighted_sum_noise;
/// let variances = vec![Variance(f64::powi(2., -30)), Variance(f64::powi(2., -32))];
/// let weights: Vec<u64> = vec![20, 10];
/// let var_out = estimate_weighted_sum_noise(&variances, &weights);
/// ```
pub fn estimate_weighted_sum_noise<T, D>(dispersion_list: &[D], weights: &[T]) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;

    for (dispersion, &w) in dispersion_list.iter().zip(weights) {
        var_res += estimate_integer_plaintext_multiplication_noise(*dispersion, w).get_variance();
    }
    Variance::from_variance(var_res)
}

/// Computes the dispersion of a multiplication
/// between an RLWE ciphertext and a scalar polynomial.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::estimate_polynomial_plaintext_multiplication_noise;
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rlwe = Variance(f64::powi(2., -40));
/// let scalar_polynomial = vec![10, 15, 18];
/// let var_out = estimate_polynomial_plaintext_multiplication_noise::<u64, _>(
///     dispersion_rlwe,
///     &scalar_polynomial,
/// );
/// ```
pub fn estimate_polynomial_plaintext_multiplication_noise<T, D>(
    dispersion: D,
    scalar_polynomial: &[T],
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    estimate_weighted_sum_noise(
        &vec![dispersion; scalar_polynomial.len()],
        scalar_polynomial,
    )
}

/// Computes the dispersion of a tensor product between two independent
/// GLWEs given a set of parameters.
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_tensor_product_noise;
/// let dimension = GlweDimension(3);
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rlwe_0 = Variance::from_modular_variance::<u64>(f64::powi(2., 24));
/// let dispersion_rlwe_1 = Variance::from_modular_variance::<u64>(f64::powi(2., 24));
/// let delta_1 = f64::powi(2., 40);
/// let delta_2 = f64::powi(2., 42);
/// let max_msg_1 = 15.;
/// let max_msg_2 = 7.;
/// let var_out = estimate_tensor_product_noise::<u64, _, _, BinaryKeyKind>(
///     polynomial_size,
///     dimension,
///     dispersion_rlwe_0,
///     dispersion_rlwe_1,
///     delta_1,
///     delta_2,
///     max_msg_1,
///     max_msg_2,
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn estimate_tensor_product_noise<T, D1, D2, K>(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: D1,
    dispersion_glwe2: D2,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    K: KeyDispersion,
{
    // constants
    let big_n = poly_size.0 as f64;
    let k = rlwe_dimension.0 as f64;
    let delta = f64::min(delta_1, delta_2);
    let delta_square = square(delta);
    let q_square = f64::powi(2., (2 * T::BITS) as i32);
    // #1
    let res_1 = big_n / delta_square
        * (dispersion_glwe1.get_modular_variance::<T>() * square(delta_2) * square(max_msg_2)
            + dispersion_glwe2.get_modular_variance::<T>() * square(delta_1) * square(max_msg_1)
            + dispersion_glwe1.get_modular_variance::<T>()
                * dispersion_glwe2.get_modular_variance::<T>());

    // #2
    let res_2 = (
        // 1ere parenthese
        (q_square - 1.) / 12.
            * (1.
                + k * big_n * K::variance_key_coefficient::<T>().get_modular_variance::<T>()
                + k * big_n * square(K::expectation_key_coefficient()))
            + k * big_n / 4. * K::variance_key_coefficient::<T>().get_modular_variance::<T>()
            + 1. / 4. * square(1. + k * big_n * K::expectation_key_coefficient())
    ) * (
        // 2e parenthese
        dispersion_glwe1.get_modular_variance::<T>() + dispersion_glwe2.get_modular_variance::<T>()
    ) * big_n
        / delta_square;

    // #3
    let res_3 = 1. / 12.
        + k * big_n / (12. * delta_square)
            * ((delta_square - 1.)
                * (K::variance_key_coefficient::<T>().get_modular_variance::<T>()
                    + square(K::expectation_key_coefficient()))
                + 3. * K::variance_key_coefficient::<T>().get_modular_variance::<T>())
        + k * (k - 1.) * big_n / (24. * delta_square)
            * ((delta_square - 1.)
                * (K::variance_coefficient_in_polynomial_key_times_key::<T>(poly_size)
                    .get_modular_variance::<T>()
                    + K::square_expectation_mean_in_polynomial_key_times_key(poly_size))
                + 3. * K::variance_coefficient_in_polynomial_key_times_key::<T>(poly_size)
                    .get_modular_variance::<T>())
        + k * big_n / (24. * delta_square)
            * ((delta_square - 1.)
                * (K::variance_odd_coefficient_in_polynomial_key_squared::<T>(poly_size)
                    .get_modular_variance::<T>()
                    + K::variance_even_coefficient_in_polynomial_key_squared::<T>(poly_size)
                        .get_modular_variance::<T>()
                    + 2. * K::squared_expectation_mean_in_polynomial_key_squared::<T>(poly_size))
                + 3. * (K::variance_odd_coefficient_in_polynomial_key_squared::<T>(poly_size)
                    .get_modular_variance::<T>()
                    + K::variance_even_coefficient_in_polynomial_key_squared::<T>(poly_size)
                        .get_modular_variance::<T>()));

    Variance::from_modular_variance::<T>(res_2 + res_1 + res_3)
}

/// Computes the dispersion of a GLWE after relinearization.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_relinearization_noise;
/// let dimension = GlweDimension(3);
/// let l_gadget = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rlk = Variance(f64::powi(2., -38));
/// let var_cmux = estimate_relinearization_noise::<u64, _, BinaryKeyKind>(
///     polynomial_size,
///     dimension,
///     dispersion_rlk,
///     base_log,
///     l_gadget,
/// );
/// ```
pub fn estimate_relinearization_noise<T, D, K>(
    poly_size: PolynomialSize,
    glwe_dimension: GlweDimension,
    dispersion_rlk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
    K: KeyDispersion,
{
    // constants
    let big_n = poly_size.0 as f64;
    let k = glwe_dimension.0 as f64;
    let base = f64::powi(2., base_log.0 as i32);
    let q_square = f64::powi(2., (2 * T::BITS) as i32);

    // first term
    let res_1 =
        k * (level.0 as f64) * big_n * dispersion_rlk.get_modular_variance::<T>() * (k + 1.) / 2.
            * (square(base) + 2.)
            / 12.;

    // second term
    let res_2 = k * big_n / 2.
        * (q_square / (12. * f64::powi(base, (2 * level.0) as i32)) - 1. / 12.)
        * ((k - 1.)
            * (K::variance_coefficient_in_polynomial_key_times_key::<T>(poly_size)
                .get_modular_variance::<T>()
                + K::square_expectation_mean_in_polynomial_key_times_key(poly_size))
            + K::variance_odd_coefficient_in_polynomial_key_squared::<T>(poly_size)
                .get_modular_variance::<T>()
            + K::variance_even_coefficient_in_polynomial_key_squared::<T>(poly_size)
                .get_modular_variance::<T>()
            + 2. * K::square_expectation_mean_in_polynomial_key_times_key(poly_size));

    // third term
    let res_3 = k * big_n / 8.
        * ((k - 1.)
            * K::variance_coefficient_in_polynomial_key_times_key::<T>(poly_size)
                .get_modular_variance::<T>()
            + K::variance_odd_coefficient_in_polynomial_key_squared::<T>(poly_size)
                .get_modular_variance::<T>()
            + K::variance_even_coefficient_in_polynomial_key_squared::<T>(poly_size)
                .get_modular_variance::<T>());

    Variance::from_modular_variance::<T>(res_1 + res_2 + res_3)
}

/// Computes the dispersion of a GLWE multiplication between two GLWEs (i.e., a
/// tensor product followed by a relinearization).
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_multiplication_noise;
/// let dimension = GlweDimension(3);
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rlwe_0 = Variance::from_modular_variance::<u64>(f64::powi(2., 24));
/// let dispersion_rlwe_1 = Variance::from_modular_variance::<u64>(f64::powi(2., 24));
/// let delta_1 = f64::powi(2., 40);
/// let delta_2 = f64::powi(2., 42);
/// let max_msg_1 = 15.;
/// let max_msg_2 = 7.;
/// let l_gadget = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let dispersion_rlk = Variance(f64::powi(2., -38));
/// let var_out = estimate_multiplication_noise::<u64, _, _, _, BinaryKeyKind>(
///     polynomial_size,
///     dimension,
///     dispersion_rlwe_0,
///     dispersion_rlwe_1,
///     delta_1,
///     delta_2,
///     max_msg_1,
///     max_msg_2,
///     dispersion_rlk,
///     base_log,
///     l_gadget,
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn estimate_multiplication_noise<T, D1, D2, D3, K>(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: D1,
    dispersion_glwe2: D2,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: D3,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    D3: DispersionParameter,
    K: KeyDispersion,
{
    // res 1
    let res_1: Variance = estimate_tensor_product_noise::<T, _, _, K>(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );

    // res 2
    let res_2: Variance = estimate_relinearization_noise::<T, _, K>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );

    Variance::from_modular_variance::<T>(
        res_1.get_modular_variance::<T>() + res_2.get_modular_variance::<T>(),
    )
}

/// Computes the dispersion of a modulus switching of an LWE encrypted with binary keys.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::LweDimension;
/// use concrete_npe::estimate_modulus_switching_noise_with_binary_key;
/// let lwe_mask_size = LweDimension(630);
/// let number_of_most_significant_bit: usize = 4;
/// let dispersion_input = Variance(f64::powi(2., -40));
/// let var_out = estimate_modulus_switching_noise_with_binary_key::<u64, _>(
///     lwe_mask_size,
///     number_of_most_significant_bit,
///     dispersion_input,
/// );
/// ```
pub fn estimate_modulus_switching_noise_with_binary_key<T, D>(
    lwe_mask_size: LweDimension,
    nb_msb: usize,
    var_in: D,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let w = (1 << nb_msb) as f64;
    let n = lwe_mask_size.0 as f64;
    let q_square = f64::powi(2., (2 * T::BITS) as i32);
    Variance::from_modular_variance::<T>(
        var_in.get_modular_variance::<T>() + 1. / 12. * q_square / square(w) - 1. / 12.
            + n / 24. * q_square / square(w)
            + n / 48.,
    )
}

/// Computes the dispersion of the constant terms of a GLWE after an LWE
/// to GLWE keyswitch.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
/// };
/// use concrete_npe::estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms;
/// let lwe_mask_size = LweDimension(630);
/// let l_ks = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let dispersion_lwe = Variance(f64::powi(2., -38));
/// let dispersion_ks = Variance(f64::powi(2., -40));
/// let var_ks = estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<u64, _, _,
/// BinaryKeyKind>(
///     lwe_mask_size,
///     dispersion_lwe,
///     dispersion_ks,
///     base_log,
///     l_ks,
/// );
/// ```
pub fn estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms<T, D1, D2, K>(
    lwe_mask_size: LweDimension,
    dispersion_lwe: D1,
    dispersion_ksk: D2,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    K: KeyDispersion,
{
    let n = lwe_mask_size.0 as f64;
    let base = (1 << base_log.0) as f64;
    let q_square = f64::powi(2., (2 * T::BITS) as i32);

    // res 1
    let res_1 = dispersion_lwe.get_modular_variance::<T>();

    // res 2
    let res_2 = n
        * (q_square / (12. * f64::powi(base, 2 * level.0 as i32)) - 1. / 12.)
        * (K::variance_key_coefficient::<T>().get_modular_variance::<T>()
            + square(K::expectation_key_coefficient()));

    // res 3
    let res_3 = n / 4. * K::variance_key_coefficient::<T>().get_modular_variance::<T>();

    // res 4
    let res_4 =
        n * (level.0 as f64) * dispersion_ksk.get_modular_variance::<T>() * (square(base) + 2.)
            / 12.;

    Variance::from_modular_variance::<T>(res_1 + res_2 + res_3 + res_4)
}

/// Computes the dispersion of the non-constant GLWE terms after an LWE to GLWE keyswitch.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
/// };
/// use concrete_npe::estimate_keyswitch_noise_lwe_to_glwe_with_non_constant_terms;
/// let lwe_mask_size = LweDimension(630);
/// let l_ks = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let dispersion_ks = Variance(f64::powi(2., -40));
/// // Compute the noise
/// let var_ks = estimate_keyswitch_noise_lwe_to_glwe_with_non_constant_terms::<u64, _>(
///     lwe_mask_size,
///     dispersion_ks,
///     base_log,
///     l_ks,
/// );
/// ```
pub fn estimate_keyswitch_noise_lwe_to_glwe_with_non_constant_terms<T, D>(
    lwe_mask_size: LweDimension,
    dispersion_ksk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let n = lwe_mask_size.0 as f64;
    let base = (1 << base_log.0) as f64;

    let res =
        n * (level.0 as f64) * dispersion_ksk.get_modular_variance::<T>() * (square(base) + 2.)
            / 12.;

    Variance::from_modular_variance::<T>(res)
}

/// Computes the dispersion of the bits greater than $q$ after a modulus switching.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::estimate_msb_noise_rlwe;
/// use std::fmt::Binary;
/// let rlwe_mask_size = PolynomialSize(1024);
/// let var_out = estimate_msb_noise_rlwe::<u64, BinaryKeyKind>(rlwe_mask_size);
/// ```
pub fn estimate_msb_noise_rlwe<T, K>(poly_size: PolynomialSize) -> Variance
where
    T: UnsignedInteger,
    K: KeyDispersion,
{
    let q_square = f64::powi(2., (2 * T::BITS) as i32);

    Variance::from_modular_variance::<T>(
        1. / q_square
            * ((q_square - 1.) / 12.
                * (1.
                    + (poly_size.0 as f64)
                        * K::variance_key_coefficient::<T>().get_modular_variance::<T>()
                    + (poly_size.0 as f64) * square(K::expectation_key_coefficient()))
                + (poly_size.0 as f64) / 4.
                    * K::variance_key_coefficient::<T>().get_modular_variance::<T>()),
    )
}

/// Computes the dispersion of an external product (between and RLWE and a GGSW)
/// encrypting a binary keys (i.e., as in TFHE PBS).
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_external_product_noise_with_binary_ggsw;
/// let poly_size = PolynomialSize(1024);
/// let mask_size = GlweDimension(2);
/// let level = DecompositionLevelCount(4);
/// let dispersion_rlwe = Variance(f64::powi(2., -40));
/// let dispersion_rgsw = Variance(f64::powi(2., -40));
/// let base_log = DecompositionBaseLog(7);
/// let var_ks = estimate_external_product_noise_with_binary_ggsw::<u64, _, _, BinaryKeyKind>(
///     poly_size,
///     mask_size,
///     dispersion_rlwe,
///     dispersion_rgsw,
///     base_log,
///     level,
/// );
/// ```
pub fn estimate_external_product_noise_with_binary_ggsw<T, D1, D2, K>(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: D1,
    var_ggsw: D2,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    K: KeyDispersion,
{
    let l = level.0 as f64;
    let k = rlwe_mask_size.0 as f64;
    let big_n = poly_size.0 as f64;
    let b = (1 << base_log.0) as f64;
    let b2l = f64::powf(b, 2. * l);

    let res_1 =
        l * (k + 1.) * big_n * var_ggsw.get_modular_variance::<T>() * (square(b) + 2.) / 12.;
    let res_2 = var_glwe.get_modular_variance::<T>() / 2.;
    let res_3 = (square(f64::powi(2., T::BITS as i32)) as f64 - b2l) / (24. * b2l)
        * (1.
            + k * big_n
                * (K::variance_key_coefficient::<T>().get_modular_variance::<T>()
                    + square(K::expectation_key_coefficient())));
    let res_4 = k * big_n / 8. * K::variance_key_coefficient::<T>().get_modular_variance::<T>();
    let res_5 = 1. / 16. * square(1. - k * big_n * K::expectation_key_coefficient());
    Variance::from_modular_variance::<T>(res_1 + res_2 + res_3 + res_4 + res_5)
}

/// Computes the dispersion of a CMUX controlled with a GGSW encrypting binary keys.
/// # Example
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_cmux_noise_with_binary_ggsw;
/// let dimension = GlweDimension(3);
/// let l_gadget = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rgsw = Variance::from_modular_variance::<u64>(f64::powi(2., 26));
/// let dispersion_rlwe_0 = Variance::from_modular_variance::<u64>(f64::powi(2., 25));
/// let dispersion_rlwe_1 = Variance::from_modular_variance::<u64>(f64::powi(2., 25));
/// // Compute the noise
/// let var_cmux = estimate_cmux_noise_with_binary_ggsw::<u64, _, _, _, BinaryKeyKind>(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     dispersion_rlwe_0,
///     dispersion_rlwe_1,
///     dispersion_rgsw,
/// );
/// ```
pub fn estimate_cmux_noise_with_binary_ggsw<T, D1, D2, D3, K>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: D1,
    dispersion_rlwe_1: D2,
    dispersion_rgsw: D3,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    D3: DispersionParameter,
    K: KeyDispersion,
{
    let var_external_product = estimate_external_product_noise_with_binary_ggsw::<T, _, _, K>(
        polynomial_size,
        dimension,
        estimate_addition_noise::<T, _, _>(dispersion_rlwe_0, dispersion_rlwe_1),
        dispersion_rgsw,
        base_log,
        l_gadget,
    );
    estimate_addition_noise::<T, _, _>(var_external_product, dispersion_rlwe_0)
}

/// Computes the dispersion of a PBS *a la TFHE* (i.e., the GGSW encrypts a
/// binary keys, and the initial noise for the RLWE is equal to zero).
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::key_kinds::BinaryKeyKind;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
/// };
/// use concrete_npe::estimate_pbs_noise;
/// let poly_size = PolynomialSize(1024);
/// let mask_size = LweDimension(2);
/// let rlwe_mask_size = GlweDimension(2);
/// let level = DecompositionLevelCount(4);
/// let dispersion_rgsw = Variance(f64::powi(2., -40));
/// let base_log = DecompositionBaseLog(7);
/// let var_ks = estimate_pbs_noise::<u64, _, BinaryKeyKind>(
///     mask_size,
///     poly_size,
///     rlwe_mask_size,
///     base_log,
///     level,
///     dispersion_rgsw,
/// );
/// ```
pub fn estimate_pbs_noise<T, D, K>(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    dispersion_bsk: D,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
    K: KeyDispersion,
{
    let n = lwe_mask_size.0 as f64;
    let k = rlwe_mask_size.0 as f64;
    let b = (1 << base_log.0) as f64;
    let l = level.0 as f64;
    let b2l = f64::powf(b, 2. * l) as f64;
    let big_n = poly_size.0 as f64;
    let q_square = f64::powi(2., (2 * T::BITS) as i32);

    let res_1 = n * l * (k + 1.) * big_n * (square(b) + 2.) / 12.
        * dispersion_bsk.get_modular_variance::<T>();
    let res_2 = n * (q_square - b2l) / (24. * b2l)
        * (1.
            + k * big_n
                * (K::variance_key_coefficient::<T>().get_modular_variance::<T>()
                    + square(K::expectation_key_coefficient())))
        + n * k * big_n / 8. * K::variance_key_coefficient::<T>().get_modular_variance::<T>()
        + n / 16. * square(1. - k * big_n * K::expectation_key_coefficient());
    Variance::from_modular_variance::<T>(res_1 + res_2)
}

#[cfg(test)]
mod tests_estimate_weighted_sum_noise {
    use super::estimate_weighted_sum_noise;
    use crate::tools::tests::assert_float_eq;
    use concrete_commons::dispersion::{DispersionParameter, Variance};
    #[test]
    fn no_noise() {
        let weights = [1u8, 1];
        let variance_in = [Variance(0.0), Variance(0.0)];
        let variance_out = estimate_weighted_sum_noise(&variance_in, &weights);
        assert_float_eq!(0.0, variance_out.get_variance(), eps = 0.0);
    }
    #[test]
    fn no_more_noise() {
        let weights = [1u8, 1, 1];
        let variance_in = [Variance(1.0), Variance(0.0)];
        let variance_out = estimate_weighted_sum_noise(&variance_in, &weights);
        assert_float_eq!(1.0, variance_out.get_variance(), eps = 0.0);
    }
    #[test]
    fn twice_the_noise() {
        let weights = [1u8, 1];
        let variance_in = [Variance(1.0), Variance(1.0)];
        let variance_out = estimate_weighted_sum_noise(&variance_in, &weights);
        assert_float_eq!(2.0, variance_out.get_variance(), eps = 0.0);
    }
    #[test]
    fn more_noise() {
        let weights = [1u8, 3];
        let variance_in = [Variance(2.0), Variance(5.0)];
        let variance_out = estimate_weighted_sum_noise(&variance_in, &weights);
        assert_float_eq!(47.0, variance_out.get_variance(), eps = 0.001);
    }
}
