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
use concrete_commons::key_kinds::{KeyKind, BinaryKeyKind};
use concrete_commons::numeric::UnsignedInteger;

/// The Gaussian secret keys have modular standard deviation set to 3.2 by default.
pub const GAUSSIAN_MODULAR_STDEV: f64 = 3.2;

pub trait KeyDistributions: KeyKind {
    ///
    ///```rust
    /// use concrete_commons::key_kinds::{KeyKind, BinaryKeyKind};
    /// use concrete_npe::KeyDistributions;
    /// BinaryKeyKind::variance_key_coefficient();
    /// ```
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance;
    fn expectation_key_coefficient() -> f64;
    // ...
}
impl KeyDistributions for BinaryKeyKind {
    fn variance_key_coefficient<T: UnsignedInteger>() -> Variance{
        Variance::from_modular_variance::<T>(1. / 4.)
    }
    fn expectation_key_coefficient() -> f64{
        0.
    }
}

/// Returns the variance of key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let var_out =
///     Variance::get_modular_variance::<ui>(&variance_key_coefficient::<ui, KeyKind::Gaussian>());
/// let expected_var_out = 10.24;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_key_coefficient<T, K>() -> Variance
where
    T: UnsignedInteger,
    K: KeyKind,
{
    match K { _ => {
        KeyKind::BinaryKeyKind => Variance::from_modular_variance::<T>(1. / 4.),
        KeyKind::TernaryKeyKind => Variance::from_modular_variance::<T>(2. / 3.),
        KeyKind::GaussianKeyKind => {
            Variance::from_modular_variance::<T>(square(GAUSSIAN_MODULAR_STDEV))
        }
        KeyKind::UniformKeyKind => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the expectation of key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_npe::expectation_key_coefficient;
///
/// let expect_out = expectation_key_coefficient::<KeyKind::Binary>();
/// let expected_expect_out = 0.5;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn expectation_key_coefficient<K>() -> f64
where
    K: KeyKind,
{
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => 1. / 2.,
        KeyKind::Ternary => 0.,
        KeyKind::Gaussian => 0.,
        KeyKind::Zero => 0.,
    }
}

/// Returns the variance of the squared key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let var_out = Variance::get_modular_variance::<ui>
/// (&variance_key_coefficient_squared::<ui, KeyKind::Ternary>());
/// let expected_var_out = 0.2222;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_key_coefficient_squared<T, K>() -> Variance
where
    T: UnsignedInteger,
    K: KeyKind,
{
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => Variance::from_modular_variance::<T>(1. / 4.),
        KeyKind::Ternary => Variance::from_modular_variance::<T>(2. / 9.),
        KeyKind::Gaussian => Variance::from_modular_variance::<T>(
            2. * square(Variance::get_modular_variance::<T>(
                &variance_key_coefficient::<T, KeyKind::Gaussian>(),
            )),
        ),
        KeyKind::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the expectation of the squared key coefficients given the key type
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let expect_out = expectation_key_coefficient_squared::<ui, KeyKind::Gaussian>();
/// let expected_expect_out = 10.24;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn expectation_key_coefficient_squared<T, K>() -> f64
where
    T: UnsignedInteger,
    K: KeyKind,
{
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => 1. / 2.,
        KeyKind::Ternary => 2. / 3.,
        KeyKind::Gaussian => {
            Variance::get_modular_variance::<T>(&variance_key_coefficient::<T, KeyKind::Gaussian>())
        }
        KeyKind::Zero => 0.,
    }
}

/// Returns the variance of the odd coefficients of a polynomial key to the
/// square given the key kind
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_odd_coefficient_in_polynomial_key_squared::<ui, KeyKind::Ternary>(
///         polynomial_size,
///     ),
/// );
/// let expected_var_out = 910.2222;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_odd_coefficient_in_polynomial_key_squared<T, K>(
    poly_size: PolynomialSize,
) -> Variance
where
    T: UnsignedInteger,
    K: KeyKind,
{
    if poly_size.0 == 1 {
        return Variance::from_modular_variance::<T>(0.);
    }
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 8.),
        KeyKind::Ternary => Variance::from_modular_variance::<T>(8. * (poly_size.0 as f64) / 9.),
        KeyKind::Gaussian => Variance::from_modular_variance::<T>(
            2. * (poly_size.0 as f64)
                * square(Variance::get_modular_variance::<T>(
                    &variance_key_coefficient::<T, KeyKind::Gaussian>(),
                )),
        ),
        KeyKind::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the variance of the even coefficients of a polynomial key to the
/// square given the key kind
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_even_coefficient_in_polynomial_key_squared::<ui, KeyKind::Binary>(
///         polynomial_size,
///     ),
/// );
/// let expected_var_out = 383.75;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_even_coefficient_in_polynomial_key_squared<T, K>(
    poly_size: PolynomialSize,
) -> Variance
where
    T: UnsignedInteger,
    K: KeyKind,
{
    if poly_size.0 == 1 {
        return Variance::from_modular_variance::<T>(
            2. * Variance::get_modular_variance::<T>(&variance_key_coefficient_squared::<T, K>()),
        );
    }
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => {
            Variance::from_modular_variance::<T>(((3 * poly_size.0 - 2) as f64) / 8.)
        }
        KeyKind::Ternary => {
            Variance::from_modular_variance::<T>(4. * ((2 * poly_size.0 - 3) as f64) / 9.)
        }
        KeyKind::Gaussian => Variance::from_modular_variance::<T>(
            2. * (poly_size.0 as f64)
                * square(Variance::get_modular_variance::<T>(
                    &variance_key_coefficient::<T, KeyKind::Gaussian>(),
                )),
        ),
        KeyKind::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key to the
/// square given the key kind
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// let expect_out = squared_expectation_mean_in_polynomial_key_squared::<ui, KeyKind::Gaussian>(
///     polynomial_size,
/// );
/// let expected_expect_out = 0.0;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn squared_expectation_mean_in_polynomial_key_squared<T, K>(poly_size: PolynomialSize) -> f64
where
    T: UnsignedInteger,
    K: KeyKind,
{
    if poly_size.0 == 1 {
        return square(expectation_key_coefficient_squared::<T, K>());
    }
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => (square(poly_size.0 as f64) + 2.) / 48.,
        KeyKind::Ternary => 0.,
        KeyKind::Gaussian => 0.,
        KeyKind::Zero => 0.,
    }
}

/// Returns the variance of the coefficients of a polynomial key resulting from
/// the multiplication of two polynomial keys of the same type (S_i x S_j)
/// given their key kind
/// # Example:
/// ```rust
/// use concrete_commons::dispersion::*;
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// type ui = u64;
/// let polynomial_size = PolynomialSize(1024);
/// //FIXME fix ZERO key kind, handle key kinds Alex implemented
/// let var_out = Variance::get_modular_variance::<ui>(
///     &variance_coefficient_in_polynomial_key_times_key::<ui, KeyKind::Zero>(polynomial_size),
/// );
/// let expected_var_out = 0.0;
/// println!("{}", var_out);
/// assert!((expected_var_out - var_out).abs() < 0.0001);
/// ```
pub fn variance_coefficient_in_polynomial_key_times_key<T, K>(poly_size: PolynomialSize) -> Variance
where
    T: UnsignedInteger,
    K: KeyKind,
{
    let key_kind: K;
    match key_kind {
        KeyKind::Binary => Variance::from_modular_variance::<T>(3. * (poly_size.0 as f64) / 16.),
        KeyKind::Ternary => Variance::from_modular_variance::<T>(4. * (poly_size.0 as f64) / 9.),
        KeyKind::Gaussian => Variance::from_modular_variance::<T>(
            square(Variance::get_modular_variance::<T>(
                &variance_key_coefficient::<T, KeyKind::Gaussian>(),
            )) * (poly_size.0 as f64),
        ),
        KeyKind::Zero => Variance::from_modular_variance::<T>(0.),
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key
/// resulting from the multiplication of two polynomial keys of the same kind
/// (S_i x S_j) given their key kind
/// # Example:
/// ```rust
/// use concrete_commons::parameters::PolynomialSize;
/// use concrete_npe::*;
///
/// let polynomial_size = PolynomialSize(2048);
/// let expect_out =
///     square_expectation_mean_in_polynomial_key_times_key::<KeyKind::Binary>(polynomial_size);
/// let expected_expect_out = 87381.375;
/// println!("{}", expect_out);
/// assert!((expected_expect_out - expect_out).abs() < 0.0001);
/// ```
pub fn square_expectation_mean_in_polynomial_key_times_key<K>(poly_size: PolynomialSize) -> f64
where
    K: KeyKind,
{
    let key_kind: K;
    match key_kind {
        KeyKind::BinaryKey => (square(poly_size.0 as f64) + 2.) / 48.,
        KeyKind::TernaryKey => 0.,
        KeyKind::GaussianKey => 0.,
        KeyKind::UniformKey => 0.,
    }
}
