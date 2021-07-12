//! Noise Propagation Estimator Module
/// * Contains material needed to estimate the growth of the noise when
///   performing homomorphic computation
use super::*;
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::{CastInto, SignedInteger, UnsignedInteger};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
/// Computes the dispersion of the error distribution after the addition of two
/// uncorrelated ciphertexts
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// let var1 = Variance::from_variance(2_f64.powf(-25.));
/// let var2 = Variance::from_variance(2_f64.powf(-25.));
/// let var_out = variance_add::<u64, _, _>(var1, var2);
/// println!("Expect Variance (2^24) =  {}", f64::powi(2., -24));
/// println!("Output Variance {}", var_out.get_variance());
/// assert!((f64::powi(2., -24) - var_out.get_variance()).abs() < 0.0001);
/// ```
pub fn variance_add<T, D1, D2>(dispersion_ct1: D1, dispersion_ct2: D2) -> Variance
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

/// Computes the dispersion of the error distribution after the addition of
/// several uncorrelated ciphertexts
/// /// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
/// let var
/// let var1 = Variance::from_variance(2_f64.powf(-25.));
/// let var2 = Variance::from_variance(2_f64.powf(-25.));
/// let var3 = Variance::from_variance(2_f64.powf(-24.));
/// let var_out = variance_add::<u64, _, _>([var1, var2, var3]);
/// println!("Expect Variance (2^24) =  {}", f64::powi(2., -24));
/// println!("Output Variance {}", var_out.get_variance());
/// assert!((f64::powi(2., -24) - var_out.get_variance()).abs() < 0.0001);
/// ```
pub fn variance_add_several<T, D>(dispersion_cts: &[D]) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;
    // The result variance is equal to the sum of the input variances
    for dispersion in dispersion_cts.iter() {
        var_res += dispersion.get_modular_variance::<T>();
    }
    Variance::from_variance(var_res)
}

//TODO: CHECK THE PRECISION !
/// Return the variance of the external product given a set of parameters.
/// To see how to use it, please refer to the test of the external product.
/// # Arguments:
/// * `dimension` - the size of the RLWE mask
/// * `l_gadget` - number of elements for the decomposition
/// * `base_log` - decomposition base of the gadget matrix
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `dispersion_rgsw` - noise dispersion of the RGSW
/// * `dispersion_rlwe` - noise dispersion of the RLWE
/// # Output:
/// * Returns the variance of the output RLWE
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::*;
/// let dimension = GlweDimension(3);
/// let polynomial_size = PolynomialSize(1024);
/// let base_log = DecompositionBaseLog(7);
/// let l_gadget = DecompositionLevelCount(4);
/// //Variance::from_variance(f64::powi(2., -38));
/// let dispersion_rgsw = Variance::from_modular_variance::<u64>(f64::powi(2., 26));
/// //Variance::from_variance(f64::powi(2., -40));
/// let dispersion_rlwe = Variance::from_modular_variance::<u64>(f64::powi(2., 24));
/// let key_type = KeyType::Binary;
/// // Computing the noise
/// let var_external_product = variance_external_product::<u64, _, _>(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     dispersion_rgsw,
///     dispersion_rlwe,
///     key_type,
/// );
/// println!("Out. {}", var_external_product.get_modular_variance::<u64>());
/// println!("Exp. {}", 2419425767395747475838549./4. );
/// assert!(( 2419425767395747475838549./4.
/// - var_external_product.get_modular_variance::<u64>()).abs() < f64::powi(10., 10));
/// ```
pub fn variance_external_product<T, D1, D2>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: D1,
    dispersion_glwe: D2,
    key_type_out: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    // norm 2 of the integer polynomial hidden in the RGSW
    // for an external product inside a bootstrap, the integer polynomial is in fact
    // a constant polynomial equal to 0 or 1
    let norm_2_msg_ggsw = 1.;
    let b_g = 1 << base_log.0;
    let q_square = f64::powi(2., (2 * T::BITS) as i32);
    println!("q = {}", T::BITS);
    println!("q_square = {}", q_square);
    println!("dimension = {}", dimension.0);
    println!("l_gadget = {}", l_gadget.0);
    println!("polynomial_size = {}", polynomial_size.0);
    println!("b_g = {}", b_g);
    println!("var(ggsw) = {}", dispersion_ggsw.get_modular_variance::<T>());
    println!("var(rlwe) = {}", dispersion_glwe.get_modular_variance::<T>());

    let res_1: f64 = ((dimension.0 + 1) * l_gadget.0 * polynomial_size.0 * (b_g * b_g + 2)) as f64
        / 12.
        * dispersion_ggsw.get_modular_variance::<T>();

    let res_2: f64 = square(norm_2_msg_ggsw)
        * (dispersion_glwe.get_modular_variance::<T>()
            + (q_square - f64::powi(b_g as f64, 2 * l_gadget.0 as i32))
                / (12. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32))
                * (1.
                    + dimension.0 as f64
                        * polynomial_size.0 as f64
                        * (variance_key_coefficient(key_type_out)
                            + square(expectation_key_coefficient(key_type_out))))
            + dimension.0 as f64 * polynomial_size.0 as f64 / 4.
                * variance_key_coefficient(key_type_out));

    Variance::from_modular_variance::<T>(res_1 + res_2)
}

/// Return the variance of the cmux given a set of parameters.
/// To see how to use it, please refer to the test of the cmux.
/// Arguments:
/// * `dimension` - the size of the RLWE mask
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `base_log` - decomposition base of the gadget matrix
/// * `l_gadget` - number of elements for the decomposition
/// * `dispersion_rlwe_0` - noise dispersion of the first RLWE
/// * `dispersion_rlwe_1` - noise dispersion of the second RLWE
/// * `dispersion_rgsw` - noise dispersion of the RGSW
/// # Output
/// * Returns the variance of the output RLWE
/// # Warning
/// * Only correct for the cmux inside a bootstrap
/// # Example
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::cmux;
/// let dimension = GlweDimension(3);
/// let l_gadget = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let polynomial_size = PolynomialSize(1024);
/// let dispersion_rgsw = Variance::from_variance(f64::powi(2., -38));
/// let dispersion_rlwe_0 = Variance::from_variance(f64::powi(2., -40));
/// let dispersion_rlwe_1 = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_cmux = cmux::<u64, _, _, _>(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     dispersion_rlwe_0,
///     dispersion_rlwe_1,
///     dispersion_rgsw,
/// );
/// ```
pub fn variance_cmux<T, D1, D2, D3>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: D1,
    dispersion_rlwe_1: D2,
    dispersion_rgsw: D3,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    D3: DispersionParameter,
{
    let var_external_product = variance_external_product::<T, _, _>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rgsw,
        variance_add::<T, _, _>(dispersion_rlwe_0, dispersion_rlwe_1),
        key_type,
    );
    let var_cmux = variance_add::<T, _, _>(var_external_product, dispersion_rlwe_0);
    var_cmux
}

/// Noise formulas for the LWE ciphertext related operations
/// Those functions will be used in the lwe tests to check that
/// the noise behavior is consistent with the theory.

/// Computes the variance of the error distribution after a multiplication
/// of a ciphertext by a scalar i.e. sigma_out^2 <- n^2 * sigma^2
/// Arguments
/// * `variance` - variance of the input LWE
/// * `n` - a signed integer
/// Output
/// * the output variance
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_npe::scalar_mul;
/// let variance = Variance::from_variance(f64::powi(2., -48));
/// let n: u64 = 543;
/// // noise computation
/// let noise = variance_scalar_mul::<u64, _>(variance, n);
/// ```
pub fn variance_scalar_mul<T, D>(variance: D, n: T) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let sn = n.into_signed();
    let product: f64 = (sn * sn).cast_into();
    return Variance::from_variance(variance.get_modular_variance::<T>() * product);
}

/// Computes the variance of the error distribution after a multisum between
/// uncorrelated ciphertexts and scalar weights i.e. sigma_out^2 <-
/// \Sum_i weight_i^2 * sigma_i^2 Arguments
/// * `dispersion_list` - a slice of f64 with the error variances of all the
///   input uncorrelated ciphertexts
/// * `weights` - a slice of Torus with the input weights
/// Output
/// * the output variance
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_npe::scalar_weighted_sum;
/// let variances = vec![
///     Variance::from_variance(f64::powi(2., -30)),
///     Variance::from_variance(f64::powi(2., -32)),
/// ];
/// let weights: Vec<u64> = vec![20, 10];
/// // noise computation
/// let noise = variance_scalar_weighted_sum(&variances, &weights);
/// ```
pub fn variance_scalar_weighted_sum<T, D>(dispersion_list: &[D], weights: &[T]) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;

    for (dispersion, &w) in dispersion_list.iter().zip(weights) {
        var_res += variance_scalar_mul(*dispersion, w).get_modular_variance::<T>();
    }
    Variance::from_variance(var_res)
}

/// Noise formulas for the RLWE operations considering that all slot have the
/// same error variance

/// Computes the variance of the error distribution after a multiplication
/// between an RLWE ciphertext and a scalar polynomial
/// sigma_out^2 <- \Sum_i weight_i^2 * sigma^2 Arguments
/// * `dispersion` - the error dispersion in each slot of the input ciphertext
/// * `scalar_polynomial` - a slice of Torus with the input weights
/// Output
/// * the error variance for each slot of the output ciphertext
pub fn variance_scalar_polynomial_mul<T, S, D>(dispersion: D, scalar_polynomial: &[T]) -> Variance
where
    T: UnsignedInteger,
    S: SignedInteger,
    D: DispersionParameter,
{
    variance_scalar_weighted_sum(
        &vec![dispersion; scalar_polynomial.len()],
        scalar_polynomial,
    )
}

/* TO REMOVE */
/* pub fn bootstrap<T, D>(
   lwe_dimension: LweDimension,
   rlwe_dimension: GlweDimension,
   polynomial_size: PolynomialSize,
   base_log: DecompositionBaseLog,
   l_gadget: DecompositionLevelCount,
   var_bsk: D,
*/

//TODO: CHECK THE FORMULA FOR THE q_square for this point
/* #[test]
pub fn test_val_qsquare() {
    val_qsquare(32);
    assert_eq!(0, 1);
}

pub fn val_qsquare<T>(q: T) {
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let q_square2 = 2_f64.powi(2 * T::BITS as i32 * 8);
    let q_square_true = square(q);
    println!("q_square = {} ", q_square);
    println!("q_square2 = {} ", q_square2);
    //println!("q_square_true = :{} ", q_square_true);
} */
/***** END of the TO REMOVE */

//TODO: IS THIS REALLY THE GET VARIANCE OR GET_VARIANCE_MODULAR ?
/// Return the variance of the tensor product between two independent GLWE given
/// a set of parameters.x. Arguments:
/// * `dimension` - the size of the RLWE mask
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `base_log` - decomposition base of the gadget matrix
/// * `l_gadget` - number of elements for the decomposition
/// * `dispersion_rlwe_0` - noise dispersion of the first RLWE
/// * `dispersion_rlwe_1` - noise dispersion of the second RLWE
/// * `dispersion_rgsw` - noise dispersion of the RGSW
/// # Output
/// * Returns the variance of the output RLWE
/// returns a variance when computing a tensorial product between two
/// independant GLWE, and rescaling it with a factor input -> var_1 = 2^22 <=>
/// std_dev lwe estimator 2^-53
pub fn variance_glwe_tensor_product_rescale_round<T, D1, D2>(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: D1,
    dispersion_glwe2: D2,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    // constants
    let big_n = poly_size.0 as f64; //TODO: polysize is defined as N+1
    let k = rlwe_dimension.0 as f64;
    let delta = f64::min(delta_1, delta_2);
    let delta_square = square(delta);
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    // #1
    let res_1 = big_n / delta_square
        * (dispersion_glwe1.get_variance() * square(delta_2) * square(max_msg_2)
            + dispersion_glwe2.get_variance() * square(delta_1) * square(max_msg_1)
            + dispersion_glwe1.get_variance() * dispersion_glwe2.get_variance());

    // #2
    let res_2 = (
        // 1ere parenthese
        (q_square - 1.) / 12.
            * (1.
                + k * big_n * variance_key_coefficient(key_type) as f64
                + k * big_n * square(expectation_key_coefficient(key_type)))
            + k * big_n / 4. * variance_key_coefficient(key_type)
            + 1. / 4. * square(1. + k * big_n * expectation_key_coefficient(key_type))
    ) * (
        // 2e parenthese
        dispersion_glwe1.get_variance() + dispersion_glwe2.get_variance()
    ) * big_n
        / delta_square;

    // #3
    let res_3 = 1. / 12.
        + k * big_n / (12. * delta_square)
            * ((delta_square - 1.)
                * (variance_key_coefficient(key_type)
                    + square(expectation_key_coefficient(key_type)))
                + 3. * variance_key_coefficient(key_type))
        + k * (k - 1.) * big_n / (24. * delta_square)
            * ((delta_square - 1.)
                * (variance_coefficient_in_polynomial_key_times_key(poly_size, key_type)
                    + square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type))
                + 3. * variance_coefficient_in_polynomial_key_times_key(poly_size, key_type))
        + k * big_n / (24. * delta_square)
            * ((delta_square - 1.)
                * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
                    + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)
                    + 2. * squared_expectation_mean_in_polynomial_key_squared(
                        poly_size, key_type,
                    ))
                + 3. * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
                    + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)));

    Variance::from_variance(res_2 + res_1 + res_3)
}

//TODO: DOC
pub fn variance_glwe_relinearization<T, D>(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    key_type: KeyType,
    dispersion_rlk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    // constants
    let big_n = poly_size.0 as f64;
    let k = mask_size.0 as f64;
    let b = f64::powi(2., base_log.0 as i32);
    let q_square = f64::powi(2., (2 * T::BITS) as i32);

    // first term
    let res_1 = k * (level.0 as f64) * big_n * dispersion_rlk.get_variance() * (k + 1.) / 2.
        * (square(b) + 2.)
        / 12.;

    // second term
    let res_2 = k * big_n / 2.
        * (q_square / (12. * f64::powi(b, (2 * level.0) as i32)) - 1. / 12.)
        * ((k - 1.)
            * (variance_coefficient_in_polynomial_key_times_key(poly_size, key_type)
                + square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type))
            + variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + 2. * square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type));

    // third term
    let res_3 = k * big_n / 8.
        * ((k - 1.) * variance_coefficient_in_polynomial_key_times_key(poly_size, key_type)
            + variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type));

    Variance::from_variance(res_1 + res_2 + res_3)
}

/// returns a variance when computing an GLWE multiplication (tensor product +
/// relinearization) input -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn variance_glwe_mul_with_relinearization<T, D1, D2, D3>(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: D1,
    dispersion_glwe2: D2,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    key_type: KeyType,
    dispersion_rlk: D3,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
    D3: DispersionParameter,
{
    // res 1
    let res_1: Variance = variance_glwe_tensor_product_rescale_round::<T, _, _>(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        key_type,
    );

    // res 2
    let res_2: Variance = variance_glwe_relinearization::<T, _>(
        poly_size,
        mask_size,
        key_type,
        dispersion_rlk,
        base_log,
        level,
    );

    // return
    Variance::from_variance(res_1.get_variance() + res_2.get_variance())
}

/// returns a variance of the drift of the PBS with binary keys
/// output -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
/// in TFHE's case nb_msb = log2(poly_size) + 1
pub fn variance_lwe_drift_pbs_with_binary_key<T, D>(
    lwe_mask_size: usize,
    nb_msb: usize,
    var_in: D,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let w = (1 << nb_msb) as f64;
    let n = lwe_mask_size as f64;
    let q_square = square(T::BITS) as f64;
    Variance::from_variance(
        square(w) * var_in.get_variance() / q_square + 1. / 12. - square(w) / (12. * q_square)
            + n / 24.
            + n * square(w) / (48. * q_square),
    )
}

/// returns a variance of the constant term of the GLWE after an LWE to GLWE key
/// switch output -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn variance_keyswitch_lwe_to_glwe_constant_term<T, D1, D2>(
    lwe_mask_size: LweDimension,
    dispersion_lwe: D1,
    dispersion_ksk: D2,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    lwe_key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    let n = lwe_mask_size.0 as f64;
    let base = (1 << base_log.0) as f64;
    let q_square = square(T::BITS) as f64;

    // res 1
    let res_1 = dispersion_lwe;

    // res 2
    let res_2 = n
        * (q_square / (12. * f64::powi(base, 2 * level.0 as i32)) - 1. / 12.)
        * (variance_key_coefficient(lwe_key_type)
            + square(expectation_key_coefficient(lwe_key_type)));

    // res 3
    let res_3 = n / 4. * variance_key_coefficient(lwe_key_type);

    // res 4
    let res_4 = n * (level.0 as f64) * dispersion_ksk.get_variance() * (square(base) + 2.) / 12.;

    // return
    Variance::from_variance(res_1.get_variance() + res_2 + res_3 + res_4)
}

/// returns a variance of the non constant GLWE after an LWE to GLWE key switch
/// output -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn variance_keyswitch_lwe_to_glwe_non_constant_terms<D>(
    lwe_mask_size: LweDimension,
    dispersion_ksk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance
where
    D: DispersionParameter,
{
    let n = lwe_mask_size.0 as f64;
    let base = (1 << base_log.0) as f64;

    // res
    let res = n * (level.0 as f64) * dispersion_ksk.get_variance() * (square(base) + 2.) / 12.;

    // return
    Variance::from_variance(res)
}

/// returns a variance of U when doing a modulus switching
/// input -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn variance_rlwe_k_1_var_u_mod_switch<T>(poly_size: usize, key_type: KeyType) -> Variance
where
    T: UnsignedInteger,
{
    let q_square = square(T::BITS) as f64;

    Variance::from_variance(
        1. / q_square
            * ((q_square - 1.) / 12.
                * (1.
                    + (poly_size as f64) * variance_key_coefficient(key_type)
                    + (poly_size as f64) * square(expectation_key_coefficient(key_type)))
                + (poly_size as f64) / 4. * variance_key_coefficient(key_type)),
    )
}

//HERE

/// returns a variance when computing a relinarization of an RLWE resulting from
/// a tensor product: with k=2, and the secret key is (-S^2(X),S(X))
/// input -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn variance_rlwe_k_2_relinearization<T, D>(
    poly_size: PolynomialSize,
    dispersion_rlk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let basis: f64 = (1 << base_log.0) as f64;
    let big_n: f64 = poly_size.0 as f64;
    let q_square = square(T::BITS) as f64;

    // res 1
    let res_1: f64 =
        (level.0 as f64) * big_n * dispersion_rlk.get_variance() * (square(basis) + 2.) / 12.;

    // res 2
    let res_2: f64 = big_n / 2.
        * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + 2. * square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type))
        * (q_square / (12. * f64::powi(basis, 2 * level.0 as i32)) - 1. / 12.)
        + big_n / 8.
            * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
                + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type));

    // return
    Variance::from_variance(res_1 + res_2)
}

/// returns a variance when computing a relinarization of an RLWE resulting from
/// a tensor product: both input had a mask size set to k input -> var_1 = 2^22
/// <=> std_dev lwe estimator 2^-53
pub fn variance_rlwe_relinearization<T, D>(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let basis: f64 = (1 << base_log.0) as f64;
    let big_n: f64 = poly_size.0 as f64;
    let k = mask_size.0 as f64;
    let q_square = square(T::BITS) as f64;

    // res 1
    let res_1: f64 = k
        * (level.0 as f64)
        * big_n
        * dispersion_rlk.get_variance()
        * (square(basis) + 2.)
        * (k + 1.)
        / 24.;

    // res 2
    let res_2: f64 = k * big_n / 2.
        * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)
            + 2. * square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type)
            + (k - 1.)
                * (variance_coefficient_in_polynomial_key_times_key(poly_size, key_type)
                    + square_expectation_mean_in_polynomial_key_times_key(poly_size, key_type)))
        * (q_square / (12. * f64::powi(basis, 2 * level.0 as i32)) - 1. / 12.)
        + k * big_n / 8.
            * (variance_odd_coefficient_in_polynomial_key_squared(poly_size, key_type)
                + variance_even_coefficient_in_polynomial_key_squared(poly_size, key_type)
                + (k - 1.) * variance_coefficient_in_polynomial_key_times_key(poly_size, key_type));

    // return
    Variance::from_variance(res_1 + res_2)
}
/*


/// returns a variance when computing the cmux
pub fn cmux(
    poly_size: usize,
    rlwe_mask_size: usize,
    var_rlwe: f64,
    var_rgsw: f64,
    base_log: usize,
    level: usize,
    q: f64,
    key_type: char,
) -> f64 {
    let res = external_product(
        poly_size,
        rlwe_mask_size,
        2. * var_rlwe,
        var_rgsw,
        base_log,
        level,
        q,
        key_type,
    ) + var_rlwe;
    res
}
*/

//TODO: update the types
/// returns a variance when computing an external product as in TFHE's PBS
/// input -> var_1 = 2^22 <=> std_dev lwe estimator 2^-53
pub fn external_product_binary_GGSW<T, D1, D2>(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rlwe: D1,
    var_rgsw: D2,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    let l = level.0 as f64;
    let k = rlwe_mask_size.0 as f64;
    let big_n = poly_size.0 as f64;
    let b = (1 << base_log.0) as f64;
    let b2l = f64::powf(b, 2. * l);

    let res_1 =
        l * (k + 1.) * big_n * var_rgsw.get_modular_variance::<T>() * (square(b) + 2.) / 12.;
    let res_2 = var_rlwe.get_modular_variance::<T>() / 2.;
    let res_3 = (square(T::BITS) as f64 - b2l) / (24. * b2l)
        * (1.
            + k * big_n
                * (variance_key_coefficient(key_type)
                    + square(expectation_key_coefficient(key_type))));
    let res_4 = k * big_n / 8. * variance_key_coefficient(key_type);
    let res_5 = 1. / 16. * square(1. - k * big_n * expectation_key_coefficient(key_type));
    Variance::from_variance(res_1 + res_2 + res_3 + res_4 + res_5)
}

//TODO: Update type
/// returns a variance when computing TFHE's PBS
pub fn variance_tfhe_pbs<T, D>(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: D,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    //TODO Correct the types
    let var_rlwe = Variance::from_modular_variance::<T>(0.);
    Variance::from_variance(
        lwe_mask_size.0 as f64
            * external_product_binary_GGSW::<T, _, _>(
                poly_size,
                rlwe_mask_size,
                var_rlwe,
                var_rgsw,
                base_log,
                level,
                key_type,
            )
            .get_modular_variance::<T>(),
    )
}

/*
/// Return the variance of output of a bootstrap given a set of parameters.
/// To see how to use it, please refer to the test of the bootstrap.
/// Arguments
/// * `lwe_dimension` - size of the LWE mask
/// * `rlwe_dimension` - size of the RLWE mask
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `base_log` - decomposition base of the gadget matrix
/// * `l_gadget` - number of elements for the decomposition
/// * `dispersion_bsk` - dispersion of the bootstrapping key
/// # Output
/// * Returns the variance of the output RLWE
/// # Example
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
/// };
/// use concrete_npe::bootstrap;
/// let rlwe_dimension = GlweDimension(3);
/// let lwe_dimension = LweDimension(630);
/// let polynomial_size = PolynomialSize(1024);
/// let base_log = DecompositionBaseLog(7);
/// let l_gadget = DecompositionLevelCount(4);
/// let var_bsk = Variance::from_variance(f64::powi(2., -38));
/// // Computing the noise
/// let var_bootstrap = bootstrap::<u64, _>(
///     lwe_dimension,
///     rlwe_dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     var_bsk,
/// );
/// ```
//TODO: WHAT IS THIS FORMULA ?
pub fn variance_bootstrap<T, D>(
    lwe_dimension: LweDimension,
    rlwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    var_bsk: D,
) -> Variance
where
    T: UnsignedInteger,
    D: DispersionParameter,
{
    let b_g = 1 << base_log.0;
    let q_square = f64::powi(2., (2 * T::BITS) as i32);
    let var_res_1: f64 = (lwe_dimension.0
        * (rlwe_dimension.0 + 1)
        * l_gadget.0
        * polynomial_size.0
        * (b_g * b_g + 2)) as f64
        / 12.
        * var_bsk.get_modular_variance::<T>();

    let var_res_2: f64 = lwe_dimension.0 as f64
        * ((rlwe_dimension.0 * polynomial_size.0 + 2) as f64
            / (24. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32)) as f64
            + lwe_dimension.0 as f64 * (rlwe_dimension.0 * polynomial_size.0 / 48 - 1 / 12) as f64
                / q_square);

    Variance::from_variance(var_res_1 + var_res_2)
}
*/

/*
/// Return the variance of the keyswitch on a LWE ciphertext given a set of
/// parameters. To see how to use it, please refer to the test of the
/// keyswitch # Warning
/// * This function compute the noise of the keyswitch without functional
///   evaluation
/// # Arguments
/// `dimension_before` - size of the input LWE mask
/// `l_ks` - number of level max for the torus decomposition
/// `base_log` - number of bits for the base B (B=2^base_log)
/// `dispersion_ks` - dispersion of the keyswitching key
/// `dispersion_input` - dispersion of the input LWE
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
/// };
/// use concrete_npe::key_switch;
/// let dimension_before = LweDimension(630);
/// let l_ks = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let dispersion_ks = Variance::from_variance(f64::powi(2., -38));
/// let dispersion_input = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_ks = key_switch::<u64, _, _>(
///     dimension_before,
///     l_ks,
///     base_log,
///     dispersion_ks,
///     dispersion_input,
/// );
/// ```
pub fn variance_key_switch_lwe_to_lwe<T, D1, D2>(
    dimension_before: LweDimension,
    l_ks: DecompositionLevelCount,
    base_log: DecompositionBaseLog,
    dispersion_ks: D1,
    dispersion_input: D2,
) -> Variance
    where
        T: UnsignedInteger,
        D1: DispersionParameter,
        D2: DispersionParameter,
{
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let var_res_1: f64 = dimension_before.0 as f64
        * (1. / 24. * f64::powi(2.0, -2 * (base_log.0 * l_ks.0) as i32) + 1. / (48. * q_square));
    let var_res_2: f64 = dimension_before.0 as f64
        * l_ks.0 as f64
        * (f64::powi(2., 2 * base_log.0 as i32) / 12. + 1. / 6.)
        * dispersion_ks.get_modular_variance::<T>();

    let var_res: f64 = dispersion_input.get_modular_variance::<T>() + var_res_1 + var_res_2;
    Variance::from_variance(var_res)
}
*/
