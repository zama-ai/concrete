//! Key distributions
//!
//! In this file are contained all the distributions for the variance and the expectation of
//! coefficients of the secret keys, secret keys to the square and different secret keys
//! multiplied by themselves.
//! The distributions are given for uniform binary, uniform ternary and gaussian secret keys.
//! Zero keys are provided for debug only.
//! All formulas are assumed to work on modular representation (i.e. not Torus representation).

use concrete_commons::dispersion::*;
use concrete_commons::parameters::PolynomialSize;

use super::*;
use concrete_commons::numeric::UnsignedInteger;

/// The Gaussian secret keys have modular standard deviation set to 3.2 by default.
pub const GAUSSIAN_MODULAR_STDEV: f64 = 3.2;

/// KeyType is an enumeration on all the different key types
/// * Uniform Binary
/// * Uniform Ternary
/// * Gaussian (centered in 0 with stdev = 3.2)
/// * Zero (all key set to zero, used only for debugging)
#[derive(Clone, Copy)]
pub enum KeyType {
    Binary,
    Ternary,
    Gaussian,
    Zero,
}

/// Returns the variance of key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let var_out =
///     Variance::get_modular_variance::<ui>(&variance_key_coefficient::<ui>(KeyType::Gaussian));
/// let expected_var_out = 10.24;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_key_coefficient<T>(key_type: KeyType) -> Variance
where
    T: UnsignedInteger,
{
    match key_type {
        KeyType::Binary => Variance::from_modular_variance::<T>(1. / 4.),
        KeyType::Ternary => Variance::from_modular_variance::<T>(2. / 3.),
        KeyType::Gaussian => Variance::from_modular_variance::<T>(square(GAUSSIAN_MODULAR_STDEV)),
        KeyType::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the expectation of key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_npe::*;
///
/// let expect_out = expectation_key_coefficient(KeyType::Binary);
/// let expected_expect_out = 0.5;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn expectation_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => 1. / 2.,
        KeyType::Ternary => 0.,
        KeyType::Gaussian => 0.,
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the squared key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let var_out = Variance::get_modular_variance::<ui>(&variance_key_coefficient_squared::<ui>(
///     KeyType::Ternary,
/// ));
/// let expected_var_out = 0.2222;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_key_coefficient_squared<T>(key_type: KeyType) -> Variance
where
    T: UnsignedInteger,
{
    match key_type {
        KeyType::Binary => Variance::from_modular_variance::<T>(1. / 4.),
        KeyType::Ternary => Variance::from_modular_variance::<T>(2. / 9.),
        KeyType::Gaussian => Variance::from_modular_variance::<T>(
            2. * square(Variance::get_modular_variance::<T>(
                &variance_key_coefficient::<T>(KeyType::Gaussian),
            )),
        ),
        KeyType::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the expectation of the squared key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let expect_out = expectation_key_coefficient_squared::<ui>(KeyType::Gaussian);
/// let expected_expect_out = 10.24;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn expectation_key_coefficient_squared<T>(key_type: KeyType) -> f64
where
    T: UnsignedInteger,
{
    match key_type {
        KeyType::Binary => 1. / 2.,
        KeyType::Ternary => 2. / 3.,
        KeyType::Gaussian => {
            Variance::get_modular_variance::<T>(&variance_key_coefficient::<T>(KeyType::Gaussian))
        }
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the odd coefficients of a polynomial key to the
/// square given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_odd_coefficient_in_polynomial_key_squared::<ui>(
///         polynomial_size,
///         KeyType::Ternary,
///     ),
/// );
/// let expected_var_out = 910.2222;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_odd_coefficient_in_polynomial_key_squared<T>(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
{
    if poly_size.0 == 1 {
        return Variance::from_modular_variance::<T>(0.);
    }
    match key_type {
        KeyType::Binary => Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 8.),
        KeyType::Ternary => Variance::from_modular_variance::<T>(8. * (poly_size.0 as f64) / 9.),
        KeyType::Gaussian => Variance::from_modular_variance::<T>(
            2. * (poly_size.0 as f64)
                * square(Variance::get_modular_variance::<T>(
                    &variance_key_coefficient::<T>(KeyType::Gaussian),
                )),
        ),
        KeyType::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the variance of the even coefficients of a polynomial key to the
/// square given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_even_coefficient_in_polynomial_key_squared::<ui>(
///         polynomial_size,
///         KeyType::Binary,
///     ),
/// );
/// let expected_var_out = 383.75;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_even_coefficient_in_polynomial_key_squared<T>(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
{
    if poly_size.0 == 1 {
        return Variance::from_modular_variance::<T>(
            2. * Variance::get_modular_variance::<T>(&variance_key_coefficient_squared::<T>(
                key_type,
            )),
        );
    }
    match key_type {
        KeyType::Binary => {
            Variance::from_modular_variance::<T>(((3 * poly_size.0 - 2) as f64) / 8.)
        }
        KeyType::Ternary => {
            Variance::from_modular_variance::<T>(4. * ((2 * poly_size.0 - 3) as f64) / 9.)
        }
        KeyType::Gaussian => Variance::from_modular_variance::<T>(
            2. * (poly_size.0 as f64)
                * square(Variance::get_modular_variance::<T>(
                    &variance_key_coefficient::<T>(KeyType::Gaussian),
                )),
        ),
        KeyType::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key to the
/// square given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let expect_out = squared_expectation_mean_in_polynomial_key_squared::<ui>(
///     polynomial_size,
///     KeyType::Gaussian,
/// );
/// let expected_expect_out = 0.0;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn squared_expectation_mean_in_polynomial_key_squared<T>(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64
where
    T: UnsignedInteger,
{
    if poly_size.0 == 1 {
        return square(expectation_key_coefficient_squared::<T>(key_type));
    }
    match key_type {
        KeyType::Binary => (square(poly_size.0 as f64) + 2.) / 48.,
        KeyType::Ternary => 0.,
        KeyType::Gaussian => 0.,
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the coefficients of a polynomial key resulting from
/// the multiplication of two polynomial keys of the same type (S_i x S_j)
/// given their key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_coefficient_in_polynomial_key_times_key::<ui>(polynomial_size, KeyType::Zero),
/// );
/// let expected_var_out = 0.0;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_coefficient_in_polynomial_key_times_key<T>(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> Variance
where
    T: UnsignedInteger,
{
    match key_type {
        KeyType::Binary => Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 16.),
        KeyType::Ternary => Variance::from_modular_variance::<T>(4. * (poly_size.0 as f64) / 9.),
        KeyType::Gaussian => Variance::from_modular_variance::<T>(
            square(Variance::get_modular_variance::<T>(
                &variance_key_coefficient::<T>(KeyType::Gaussian),
            )) * (poly_size.0 as f64),
        ),
        KeyType::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key
/// resulting from the multiplication of two polynomial keys of the same type
/// (S_i x S_j) given their key type
/// # Example:
/// ```rust
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// let polynomial_size = PolynomialSize(2048);
/// let expect_out =
///     square_expectation_mean_in_polynomial_key_times_key(polynomial_size, KeyType::Binary);
/// let expected_expect_out = 87381.375;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn square_expectation_mean_in_polynomial_key_times_key(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64 {
    match key_type {
        KeyType::Binary => (square(poly_size.0 as f64) + 2.) / 48.,
        KeyType::Ternary => 0.,
        KeyType::Gaussian => 0.,
        KeyType::Zero => 0.,
    }
}
