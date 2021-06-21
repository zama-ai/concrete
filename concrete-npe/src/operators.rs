/// Noise Propagation Estimator Module
/// * Contains material needed to estimate the growth of the noise when
///   performing homomorphic computation
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::{CastInto, SignedInteger, UnsignedInteger};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};

/// Computes the dispersion of the error distribution after the addition of two
/// uncorrelated ciphertexts
/// Arguments:
/// * `dispersion_ct1` - noise dispersion of the first ciphertext
/// * `dispersion_ct2` - noise dispersion of the second ciphertext
/// Output
/// * The noise variance of the sum of the first and the second ciphertext
pub fn add<D1, D2>(dispersion_ct1: D1, dispersion_ct2: D2) -> Variance
where
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    // The result variance is equal to the sum of the input variances
    let var_res: f64 = dispersion_ct1.get_variance() + dispersion_ct2.get_variance();
    Variance::from_variance(var_res)
}

/// Computes the dispersion of the error distribution after the addition of several
/// uncorrelated ciphertexts
/// Argument:
/// * `dispersion_cts` - noise variance of the ciphertexts
/// Output:
/// * the noise variance of the sum of the ciphertexts
pub fn add_several<D>(dispersion_cts: &[D]) -> Variance
where
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;
    // The result variance is equal to the sum of the input variances
    for dispersion in dispersion_cts.iter() {
        var_res += dispersion.get_variance();
    }
    Variance::from_variance(var_res)
}

/// Return the variance of the external product given a set of parameters.
/// To see how to use it, please refer to the test of the external product.
/// Arguments:
/// * `dimension` - the size of the RLWE mask
/// * `l_gadget` - number of elements for the decomposition
/// * `base_log` - decomposition base of the gadget matrix
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
/// * `dispersion_rgsw` - noise dispersion of the RGSW
/// * `dispersion_rlwe` - noise dispersion of the RLWE
/// # Output:
/// * Returns the variance of the output RLWE
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::external_product;
/// let dimension = GlweDimension(3);
/// let polynomial_size = PolynomialSize(1024);
/// let base_log = DecompositionBaseLog(7);
/// let l_gadget = DecompositionLevelCount(4);
/// let dispersion_rgsw = Variance::from_variance(f64::powi(2., -38));
/// let dispersion_rlwe = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_external_product = external_product::<u64,_,_>(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     dispersion_rgsw,
///     dispersion_rlwe,
/// );
/// ```
pub fn external_product<T: UnsignedInteger, D1, D2>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rgsw: D1,
    dispersion_rlwe: D2,
) -> Variance
where
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    // norm 2 of the integer polynomial hidden in the RGSW
    // for an external product inside a bootstrap, the integer polynomial is in fact
    // a constant polynomial equal to 0 or 1
    let norm_2_msg_rgsw = 1.;
    let b_g = 1 << base_log.0;
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let res_1: f64 = ((dimension.0 + 1) * l_gadget.0 * polynomial_size.0 * (b_g * b_g + 2)) as f64
        / 12.
        * dispersion_rgsw.get_variance();

    let res_2: f64 = norm_2_msg_rgsw
        * ((dimension.0 * polynomial_size.0 + 2) as f64
            / (24. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32)) as f64
            + (dimension.0 * polynomial_size.0 / 48 - 1 / 12) as f64 / q_square);

    let res_3: f64 = norm_2_msg_rgsw * dispersion_rlwe.get_variance();
    Variance::from_variance(res_1 + res_2 + res_3)
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
pub fn cmux<T: UnsignedInteger, D1, D2, D3>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: D1,
    dispersion_rlwe_1: D2,
    dispersion_rgsw: D3,
) -> Variance
where
    D1: DispersionParameter,
    D2: DispersionParameter,
    D3: DispersionParameter,
{
    let var_external_product = external_product::<T, _, _>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rgsw,
        add(dispersion_rlwe_0, dispersion_rlwe_1),
    );
    let var_cmux = add(var_external_product, dispersion_rlwe_0);
    var_cmux
}

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
pub fn bootstrap<T: UnsignedInteger, V>(
    lwe_dimension: LweDimension,
    rlwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    var_bsk: V,
) -> Variance
where
    V: DispersionParameter,
{
    let b_g = 1 << base_log.0;
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let var_res_1: f64 = (lwe_dimension.0
        * (rlwe_dimension.0 + 1)
        * l_gadget.0
        * polynomial_size.0
        * (b_g * b_g + 2)) as f64
        / 12.
        * var_bsk.get_variance();

    let var_res_2: f64 = lwe_dimension.0 as f64
        * ((rlwe_dimension.0 * polynomial_size.0 + 2) as f64
            / (24. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32)) as f64
            + lwe_dimension.0 as f64 * (rlwe_dimension.0 * polynomial_size.0 / 48 - 1 / 12) as f64
                / q_square);

    Variance::from_variance(var_res_1 + var_res_2)
}

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
/// let var_ks = key_switch::<u64, _, _>(dimension_before, l_ks, base_log, dispersion_ks, dispersion_input);
/// ```
pub fn key_switch<T: UnsignedInteger, D1, D2>(
    dimension_before: LweDimension,
    l_ks: DecompositionLevelCount,
    base_log: DecompositionBaseLog,
    dispersion_ks: D1,
    dispersion_input: D2,
) -> Variance
where
    D1: DispersionParameter,
    D2: DispersionParameter,
{
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let var_res_1: f64 = dimension_before.0 as f64
        * (1. / 24. * f64::powi(2.0, -2 * (base_log.0 * l_ks.0) as i32) + 1. / (48. * q_square));
    let var_res_2: f64 = dimension_before.0 as f64
        * l_ks.0 as f64
        * (f64::powi(2., 2 * base_log.0 as i32) / 12. + 1. / 6.)
        * dispersion_ks.get_variance();

    let var_res: f64 = dispersion_input.get_variance() + var_res_1 + var_res_2;
    Variance::from_variance(var_res)
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
/// let noise = scalar_mul(variance, n);
/// ```
pub fn scalar_mul<T, V>(variance: V, n: T) -> Variance
where
    T: UnsignedInteger,
    V: DispersionParameter,
{
    let sn = n.into_signed();
    let product: f64 = (sn * sn).cast_into();
    return Variance::from_variance(variance.get_variance() * product);
}

/// Computes the variance of the error distribution after a multisum between
/// uncorrelated ciphertexts and scalar weights i.e. sigma_out^2 <-
/// \Sum_i weight_i^2 * sigma_i^2 Arguments
/// * `dispersion_list` - a slice of f64 with the error variances of all the input
///   uncorrelated ciphertexts
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
/// let noise = scalar_weighted_sum(&variances, &weights);
/// ```
pub fn scalar_weighted_sum<T: UnsignedInteger, D>(dispersion_list: &[D], weights: &[T]) -> Variance
where
    D: DispersionParameter,
{
    let mut var_res: f64 = 0.;

    for (dispersion, &w) in dispersion_list.iter().zip(weights) {
        var_res += scalar_mul(*dispersion, w).get_variance();
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
pub fn scalar_polynomial_mult<T: UnsignedInteger, S: SignedInteger, D>(
    dispersion: D,
    scalar_polynomial: &[T],
) -> Variance
where
    D: DispersionParameter,
{
    scalar_weighted_sum(
        &vec![dispersion; scalar_polynomial.len()],
        scalar_polynomial,
    )
}
