use concrete_commons::parameters::PolynomialSize;

use super::*;

pub const GAUSSIAN_STDEV: f64 = 3.2 / (1_u128 << 64) as f64;

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
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The noise variance of the coefficients of the key
pub fn variance_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => 1. / 4.,
        KeyType::Ternary => 2. / 3.,
        KeyType::Gaussian => square(GAUSSIAN_STDEV * f64::powi(2., 64 as i32)),
        KeyType::Zero => 0.,
    }
}

/// Returns the expectation of key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The expectation of the coefficients of the key
pub fn expectation_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => 1. / 2.,
        KeyType::Ternary => 0.,
        KeyType::Gaussian => 0.,
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the squared key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The noise variance of the squared coefficients of the key
pub fn variance_key_coefficient_squared(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => 1. / 4.,
        KeyType::Ternary => 2. / 9.,
        KeyType::Gaussian => 2. * square(variance_key_coefficient(KeyType::Gaussian)),
        KeyType::Zero => 0.,
    }
}

/// Returns the expectation of the squared key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The expectation of the squared coefficients of the key
pub fn expectation_key_coefficient_squared(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => 1. / 2.,
        KeyType::Ternary => 2. / 3.,
        KeyType::Gaussian => variance_key_coefficient(KeyType::Gaussian),
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the odd coefficients of a polynomial key to the
/// square given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The noise variance of the odd coefficients of a polynomial key to the
///   square
pub fn variance_odd_coefficient_in_polynomial_key_squared(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64 {
    if poly_size.0 == 1 {
        return 0.;
    }
    match key_type {
        KeyType::Binary => 3. * (poly_size.0 as f64) / 8.,
        KeyType::Ternary => 8. * (poly_size.0 as f64) / 9.,
        KeyType::Gaussian => {
            2. * (poly_size.0 as f64) * square(variance_key_coefficient(KeyType::Gaussian))
        }
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the even coefficients of a polynomial key to the
/// square given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The noise variance of the even coefficients of a polynomial key to the
///   square
pub fn variance_even_coefficient_in_polynomial_key_squared(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64 {
    if poly_size.0 == 1 {
        return 2. * variance_key_coefficient_squared(key_type);
    }
    match key_type {
        KeyType::Binary => ((3 * poly_size.0 - 2) as f64) / 8.,
        KeyType::Ternary => 4. * ((2 * poly_size.0 - 3) as f64) / 9.,
        KeyType::Gaussian => {
            2. * (poly_size.0 as f64) * square(variance_key_coefficient(KeyType::Gaussian))
        }
        KeyType::Zero => 0.,
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key to the
/// square given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output:
/// * The mean expectation of the coefficients of a polynomial key to the square
pub fn squared_expectation_mean_in_polynomial_key_squared(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64 {
    if poly_size.0 == 1 {
        return square(expectation_key_coefficient_squared(key_type));
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
/// Arguments:
/// * `key_type` - input key type (both keys in the product have the same key
///   type)
/// Output:
/// * The noise variance of the coefficients of the polynomial key S_i x S_j
pub fn variance_coefficient_in_polynomial_key_times_key(
    poly_size: PolynomialSize,
    key_type: KeyType,
) -> f64 {
    match key_type {
        KeyType::Binary => 3. * (poly_size.0 as f64) / 16.,
        KeyType::Ternary => 4. * (poly_size.0 as f64) / 9.,
        KeyType::Gaussian => {
            square(variance_key_coefficient(KeyType::Gaussian)) * (poly_size.0 as f64)
        }
        KeyType::Zero => 0.,
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key
/// resulting from the multiplication of two polynomial keys of the same type
/// (S_i x S_j) given their key type
/// Arguments:
/// * `key_type` - input key type (both keys in the product have the same key
///   type)
/// Output:
/// * The mean expectation of the coefficients of the polynomial key S_i x S_j
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
