/// Noise Propagation Estimator Module
/// * Contains material needed to estimate the growth of the noise when
///   performing homomophic computation
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::{CastInto, SignedInteger, UnsignedInteger};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use itertools::izip;

/// Computes the variance of the error distribution after the addition of two
/// uncorrelated ciphertexts Arguments
/// * `var_ct1` - noise variance of the first ciphertext
/// * `var_ct2` - noise variance of the second ciphertext
/// Output
/// * the variance of the sum of the first and the second ciphertext
pub fn add_ciphertexts(var_ct1: f64, var_ct2: f64) -> f64 {
    var_ct1 + var_ct2
}

/// Computes the variance of the error distribution after the addition several
/// uncorrelated ciphertexts
/// Argument
/// * `var_cts` - noise variance of the ciphertexts
/// Output
/// * the variance of the sum of the ciphertexts
pub fn add_several_ciphertexts(var_cts: &[f64]) -> f64 {
    let mut res: f64 = 0.;
    for var in var_cts.iter() {
        res += *var;
    }
    res
}

/// Computes the number of bits affected by the noise with a variance var
/// describing a normal distribution takes into account the number of bits of
/// the integers
pub fn nb_bit_from_variance_99(var: f64, torus_bit: usize) -> usize {
    // compute sigma
    let sigma: f64 = f64::sqrt(var);

    // the constant to get 99% of the normal distribution
    let z: f64 = 3.;
    let tmp = torus_bit as f64 + f64::log2(sigma * z);
    if tmp < 0. {
        // means no bits are affected by the noise in the integer representation
        // (discrete space)
        0usize
    } else {
        tmp.ceil() as usize
    }
}

/// Return the variance of the external product given a set of parameters.
/// To see how to use it, please refer to the test of the external product.
/// Arguments
/// * `dimension` - the size of the RLWE mask
/// * `l_gadget` - number of elements for the Torus decomposition
/// * `base_log` - decomposition base of the gadget matrix
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree + 1
/// * `var_trgsw` - noise variance of the TRGSW
/// * `var_trlwe` - noise variance of the TRLWE
/// # Output
/// * Returns the variance of the output RLWE
/// # Warning
/// * only correct for the external product inside a bootstrap
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::external_product;
/// type torus = u64; // or u32
///                   // settings
/// let dimension = GlweDimension(3);
/// let polynomial_size = PolynomialSize(1024);
/// let base_log = DecompositionBaseLog(7);
/// let l_gadget = DecompositionLevelCount(4);
/// let var_trgsw = Variance::from_variance(f64::powi(2., -38));
/// let var_trlwe = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_external_product = external_product(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     var_trgsw,
///     var_trlwe,
/// );
/// ```
pub fn external_product<T: UnsignedInteger, V1, V2>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    var_trgsw: V1,
    var_trlwe: V2,
) -> Variance
where
    V1: DispersionParameter,
    V2: DispersionParameter,
{
    // norm 2 of the integer polynomial hidden in the TRGSW
    // for an external product inside a bootstrap, the integer polynomial is in fact
    // a constant polynomial equal to 0 or 1
    let norm_2_msg_trgsw = 1.;
    let b_g = 1 << base_log.0;
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let res_1: f64 = ((dimension.0 + 1) * l_gadget.0 * polynomial_size.0 * (b_g * b_g + 2)) as f64
        / 12.
        * var_trgsw.get_variance();

    let res_2: f64 = norm_2_msg_trgsw
        * ((dimension.0 * polynomial_size.0 + 2) as f64
            / (24. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32)) as f64
            + (dimension.0 * polynomial_size.0 / 48 - 1 / 12) as f64 / q_square);

    let res_3: f64 = norm_2_msg_trgsw * var_trlwe.get_variance();
    let res: Variance = Variance::from_variance(res_1 + res_2 + res_3);
    return res;
}

/// Return the variance of the cmux given a set of parameters.
/// To see how to use it, please refer to the test of the cmux.
/// Arguments
/// * `dimension` - the size of the RLWE mask
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `base_log` - decomposition base of the gadget matrix
/// * `l_gadget` - number of elements for the Torus decomposition
/// * `var_rlwe_0` - noise variance of the first TRLWE
/// * `var_rlwe_1` - noise variance of the second TRLWE
/// * `var_trgsw` - noise variance of the TRGSW
/// # Output
/// * Returns the variance of the output RLWE
/// # Warning
/// * only correct for the cmux inside a bootstrap
/// # Example
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, PolynomialSize,
/// };
/// use concrete_npe::cmux;
/// type torus = u64; // or u32
///                   // settings
/// let dimension = GlweDimension(3);
/// let l_gadget = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let polynomial_size = PolynomialSize(1024);
/// let var_trgsw = Variance::from_variance(f64::powi(2., -38));
/// let var_trlwe_0 = Variance::from_variance(f64::powi(2., -40));
/// let var_trlwe_1 = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_cmux = cmux::<torus, _, _, _>(
///     dimension,
///     polynomial_size,
///     base_log,
///     l_gadget,
///     var_trlwe_0,
///     var_trlwe_1,
///     var_trgsw,
/// );
/// ```
pub fn cmux<T: UnsignedInteger, V1, V2, V3>(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    var_rlwe_0: V1,
    var_rlwe_1: V2,
    var_trgsw: V3,
) -> Variance
where
    V1: DispersionParameter,
    V2: DispersionParameter,
    V3: DispersionParameter,
{
    let var_external_product = external_product::<T, _, _>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        var_trgsw,
        Variance::from_variance(add_ciphertexts(
            var_rlwe_0.get_variance(),
            var_rlwe_1.get_variance(),
        )),
    );
    let var_cmux = add_ciphertexts(
        var_external_product.get_variance(),
        var_rlwe_0.get_variance(),
    );
    return Variance::from_variance(var_cmux);
}

/// Return the variance of output of a bootstrap given a set of parameters.
/// To see how to use it, please refer to the test of the bootstrap.
/// Arguments
/// * `lwe_dimension` - size of the LWE mask
/// * `rlwe_dimension` - size of the RLWE mask
/// * `polynomial_size` - number of coefficients of the polynomial e.g. degree +
///   1
/// * `base_log` - decomposition base of the gadget matrix
/// * `l_gadget` - number of elements for the Torus decomposition
/// * `var_bsk` - variance of the bootstrapping key
/// # Output
/// * Returns the variance of the output RLWE
/// # Example
/// ```rust
/// use concrete_commons::dispersion::{DispersionParameter, Variance};
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
/// };
/// use concrete_npe::bootstrap;
/// type Torus = u64; // or u32
///                   // settings
/// let rlwe_dimension = GlweDimension(3);
/// let lwe_dimension = LweDimension(630);
/// let polynomial_size = PolynomialSize(1024);
/// let base_log = DecompositionBaseLog(7);
/// let l_gadget = DecompositionLevelCount(4);
/// let var_bsk = Variance::from_variance(f64::powi(2., -38));
/// // Computing the noise
/// let var_bootstrap = bootstrap(
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
    let res_1: f64 = (lwe_dimension.0
        * (rlwe_dimension.0 + 1)
        * l_gadget.0
        * polynomial_size.0
        * (b_g * b_g + 2)) as f64
        / 12.
        * var_bsk.get_variance();

    let res_2: f64 = lwe_dimension.0 as f64
        * ((rlwe_dimension.0 * polynomial_size.0 + 2) as f64
            / (24. * f64::powi(b_g as f64, 2 * l_gadget.0 as i32)) as f64
            + lwe_dimension.0 as f64 * (rlwe_dimension.0 * polynomial_size.0 / 48 - 1 / 12) as f64
                / q_square);

    let res: f64 = res_1 + res_2;
    return Variance::from_variance(res);
}

/// Computes tho variance of the error during a bootstrap due to the round on
/// the LWE mask # Argument
/// * `lwe_dimension` - size of the LWE mask
/// # Output
/// * Return the variance of the error
pub fn drift_index_lut(lwe_dimension: LweDimension) -> f64 {
    (lwe_dimension.0 as f64) / 16.0
}

#[allow(dead_code, clippy::too_many_arguments)]
pub fn bootstrap_then_key_switch(
    n: LweDimension,
    l_bsk: DecompositionLevelCount,
    log_base_bsk: DecompositionBaseLog,
    stdev_bsk: f64,
    polynomial_size: PolynomialSize,
    l_ks: DecompositionLevelCount,
    log_base_ks: DecompositionBaseLog,
    stdev_ks: f64,
) -> Variance {
    let res_1: f64 = (2 * n.0 * l_bsk.0 * polynomial_size.0 * (1 << (2 * (log_base_bsk.0 - 1))))
        as f64
        * f64::powi(stdev_bsk, 2);
    let res_2: f64 = (n.0 * (polynomial_size.0 + 1)) as f64
        * (f64::powi(2.0, -2 * (log_base_bsk.0 * l_bsk.0 + 1) as i32));
    let res_3: f64 =
        polynomial_size.0 as f64 * f64::powi(2.0, -2 * (log_base_ks.0 * l_ks.0 + 1) as i32);
    let res_4: f64 = polynomial_size.0 as f64
        * l_ks.0 as f64
        * f64::powi(2., log_base_ks.0 as i32 - 1)
        * f64::powi(stdev_ks, 2);
    let res: f64 = res_1 + res_2 + res_3 + res_4;
    return Variance::from_variance(res);
}

/// Return the variance of the keyswitch on a LWE sample given a set of
/// parameters. To see how to use it, please refer to the test of the
/// keyswitch # Warning
/// * This function compute the noise of the keyswitch without functional
///   evaluation
/// # Arguments
/// `dimension_before` - size of the input LWE mask
/// `l_ks` - number of level max for the torus decomposition
/// `base_log` - number of bits for the base B (B=2^base_log)
/// `var_ks` - variance of the keyswitching key
/// `var_input` - variance of the input LWE
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_commons::parameters::{
///     DecompositionBaseLog, DecompositionLevelCount, LweDimension,
/// };
/// use concrete_npe::key_switch;
/// let torus = u32; // or u64
///                  // settings
/// let dimension_beforei = LweDimension(630);
/// let l_ks = DecompositionLevelCount(4);
/// let base_log = DecompositionBaseLog(7);
/// let var_ks = Variance::from_variance(f64::powi(2., -38));
/// let var_input = Variance::from_variance(f64::powi(2., -40));
/// // Computing the noise
/// let var_ks = key_switch::<torus, _, _>(dimension_before, l_ks, base_log, var_ks, var_input);
/// ```
pub fn key_switch<T: UnsignedInteger, V1, V2>(
    dimension_before: LweDimension,
    l_ks: DecompositionLevelCount,
    base_log: DecompositionBaseLog,
    var_ks: V1,
    var_input: V2,
) -> Variance
where
    V1: DispersionParameter,
    V2: DispersionParameter,
{
    let q_square = f64::powi(2., (2 * T::BITS * 8) as i32);
    let res_1: f64 = dimension_before.0 as f64
        * (1. / 24. * f64::powi(2.0, -2 * (base_log.0 * l_ks.0) as i32) + 1. / (48. * q_square));
    let res_2: f64 = dimension_before.0 as f64
        * l_ks.0 as f64
        * (f64::powi(2., 2 * base_log.0 as i32) / 12. + 1. / 6.)
        * var_ks.get_variance();

    let res: f64 = var_input.get_variance() + res_1 + res_2;
    return Variance::from_variance(res);
}

/// Noise formulas for the lwe-sample related operations
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
/// use concrete_npe::single_scalar_mul;
/// let torus = u32; // or u64
///                   // parameters
/// let variance = Variance::from_variance(f64::powi(2., -48));
/// let n: torus = (-543 as i64) as torus;
/// // noise computation
/// let noise = single_scalar_mul::<torus, _>(variance, n);
/// ```
pub fn single_scalar_mul<T, V>(variance: V, n: T) -> Variance
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
/// * `variances` - a slice of f64 with the error variances of all the input
///   uncorrelated ciphertexts
/// * `weights` - a slice of Torus with the input weights
/// Output
/// * the output variance
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// use concrete_npe::multisum_uncorrelated;
/// let torus = u32; // or u64
///                  // parameters
/// let variances = vec![
///     Variance::from_variance(f64::powi(2., -30)),
///     Variance::from_variance(f64::powi(2., -32)),
/// ];
/// let weights: Vec<Torus> = vec![(-543 as i64) as Torus, 10 as Torus];
/// // noise computation
/// let noise = multisum_uncorrelated::<torus, _>(&variances, &weights);
/// ```
pub fn multisum_uncorrelated<T: UnsignedInteger, V>(variances: &[V], weights: &[T]) -> Variance
where
    V: DispersionParameter,
{
    let mut new_variance: Variance = Variance::from_variance(0.);

    for (var_ref, &w) in variances.iter().zip(weights) {
        new_variance = Variance::from_variance(
            new_variance.get_variance() + single_scalar_mul(*var_ref, w).get_variance(),
        );
    }

    return new_variance;
}

/// Computes the variance of the error distribution after a multiplication
/// of several ciphertexts by several scalars Arguments
/// * `var_out` - variances of the output LWEs (output)
/// * `var_in` - variances of the input LWEs
/// * `t` - a slice of signed integer
/// # Example
/// ```rust
/// use concrete_npe::scalar_mul ;
/// use concrete_commons::dispersion::Variance;
/// let torus = u32; // or u64
/// // parameters
/// let var_in = vec![Variance::from_variance(f64::powi(2., -30)), Variance::from_variance(f64::powi
/// (2., -32))] ;
/// let weights: Vec<Torus> = vec![(-543 as i64) as Torus, 10 as Torus] ;
/// // noise computation
/// let mut var_out = vec![Variance::from_variance(0.); 2] ;
/// scalar_mul::<torus, _>(&mut var_out, &var_in, &weights) ;
/// ```
pub fn scalar_mul<T: UnsignedInteger, V>(var_out: &mut [Variance], var_in: &[V], t: &[T])
where
    V: DispersionParameter,
{
    for (vo, vi, tval) in izip!(var_out.iter_mut(), var_in.iter(), t.iter()) {
        *vo = single_scalar_mul(*vi, *tval);
    }
}

/// Computes the variance of the error distribution after a multiplication
/// of several ciphertexts by several scalars Arguments
/// * `var` - variances of the input/output LWEs (output)
/// * `t` - a slice of signed integer
/// # Example
/// ```rust
/// use concrete_commons::dispersion::Variance;
/// let torus = u32; // or u64
///                  // parameters
/// let mut var = vec![
///     Variance::from_variance(f64::powi(2., -30)),
///     Variance::from_variance(f64::powi(2., -32)),
/// ];
/// let weights: Vec<torus> = vec![(-543 as i64) as Torus, 10 as Torus];
/// // noise computation
/// scalar_mul_inplace::<torus>(&mut var, &weights);
/// ```
fn scalar_mul_inplace<T: UnsignedInteger>(var: &mut [Variance], t: &[T]) {
    for (v, tval) in izip!(var.iter_mut(), t.iter()) {
        *v = single_scalar_mul(*v, *tval);
    }
}

/// Computes the variance of the error distribution after an addition of two
/// uncorrelated ciphertexts sigma_out^2 <- sigma0^2 + sigma1^2
/// Arguments
/// * `variance_0` - variance of the error of the first input ciphertext
/// * `variance_1` - variance of the error of the second input ciphertext
/// Output
/// * the sum of the variances
pub fn add_2_uncorrelated<V0, V1>(variance_0: V0, variance_1: V1) -> Variance
where
    V0: DispersionParameter,
    V1: DispersionParameter,
{
    Variance::from_variance(variance_0.get_variance() + variance_1.get_variance())
}

/// Computes the variance of the error distribution after an addition of n
/// uncorrelated ciphertexts sigma_out^2 <- \Sum sigma_i^2
/// Arguments
/// * `variances` - a slice of f64 with the error variances of all the input
///   uncorrelated ciphertexts
/// Output
/// * the sum of the variances
pub fn add_n_uncorrelated<V>(variances: &[V]) -> Variance
where
    V: DispersionParameter,
{
    let mut new_variance = Variance::from_variance(0.);
    for var in variances.iter() {
        new_variance = Variance::from_variance(new_variance.get_variance() + var.get_variance());
    }
    new_variance
}

/// Computes an upper bound for the number of 1 in a secret key
/// z*sigma + mean
pub fn upper_bound_hw_secret_key(n: usize) -> usize {
    let n_f: f64 = n as f64;
    let sigma: f64 = f64::sqrt(n_f) / 2.;
    let mean: f64 = n_f / 2.;
    let z: f64 = 3.;
    (mean + z * sigma).round() as usize
}

/// Computes an upper bound for the log2 of the rounding noise
/// z*sigma (mean is 0.)
/// 2048 -> 4.838859820820504
/// 1024 -> 4.357122758833062
///  630 -> 4.024243436996168
///  512 -> 3.8824357953680453
pub fn log2_rounding_noise(n: usize) -> f64 {
    let bound_sup: f64 = upper_bound_hw_secret_key(n) as f64;
    let sigma: f64 = f64::sqrt(bound_sup / 12.);
    let z: f64 = 3.;
    f64::log2(sigma * z)
}

/// Noise formulas for the RLWE operations considering that all slot have the
/// same error variance

/// Computes the variance of the error distribution after a multiplication
/// between a RLWE sample and a scalar polynomial sigma_out^2 <- \Sum_i
/// weight_i^2 * sigma^2 Arguments
/// * `variance` - the error variance in each slot of the input ciphertext
/// * `scalar_polynomial` - a slice of Torus with the input weights
/// Output
/// * the error variance for each slot of the output ciphertext
pub fn scalar_polynomial_mult<T: UnsignedInteger, S: SignedInteger, V>(
    variance: V,
    scalar_polynomial: &[T],
) -> Variance
where
    V: DispersionParameter,
{
    return multisum_uncorrelated(&vec![variance; scalar_polynomial.len()], scalar_polynomial);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_test_bootstrap_then_key_switch() {
        let n = LweDimension(800); // 630;
        let l_bsk = DecompositionLevelCount(3);
        let log_base_bsk = DecompositionBaseLog(7);
        let stdev_bsk: f64 = f64::powi(2., -25);
        let polynomial_size = PolynomialSize(1 << 10);
        let l_ks = DecompositionLevelCount(8);
        let log_base_ks = DecompositionBaseLog(3);
        let stdev_ks: f64 = f64::powi(2., -19);
        bootstrap_then_key_switch(
            n,
            l_bsk,
            log_base_bsk,
            stdev_bsk,
            polynomial_size,
            l_ks,
            log_base_ks,
            stdev_ks,
        );
    }
}
