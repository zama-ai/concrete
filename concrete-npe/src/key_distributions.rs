/// KeyType is an enumeration on all the different key types
/// * Uniform Binary
/// * Uniform Ternary
/// * Gaussian (centered in 0 with stdev = 3.2)
/// * Zero (used only for testing)
pub enum KeyType {
    Binary,
    Ternary,
    Gaussian,
    Zero,
}

/// Returns the variance of key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The noise variance of the coefficients of the key 
pub fn var_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => var_binary_key_coefficient(),
        KeyType::Ternary => var_ternary_key_coefficient(),
        KeyType::Gaussian => var_gaussian_key_coefficient(),
        KeyType::Zero => 0.,
    }
}

/// Returns the expectation of key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The expectation of the coefficients of the key 
pub fn expect_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => expect_binary_key_coefficient(),
        KeyType::Ternary => expect_ternary_key_coefficient(),
        KeyType::Gaussian => expect_gaussian_key_coefficient(),
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the squared key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The noise variance of the squared coefficients of the key 
pub fn var_coef_key_square(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => var_binary_coef_key_square(),
        KeyType::Ternary => var_ternary_coef_key_square(),
        KeyType::Gaussian => var_gaussian_coef_key_square(),
        KeyType::Zero => 0.,
    }
}

/// Returns the expectation of the squared key coefficients given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The expectation of the squared coefficients of the key 
pub fn expect_coef_key_square(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => expect_binary_coef_key_square(),
        KeyType::Ternary => expect_ternary_coef_key_square(),
        KeyType::Gaussian => expect_gaussian_coef_key_square(),
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the odd coefficients of a polynomial key to the square 
/// given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The noise variance of the odd coefficients of a polynomial key to the square 
pub fn var_odd_poly_key_square(poly_size: usize, key_type: KeyType) -> f64 {
    if poly_size == 1 {
        return 0.;
    }
    match key_type {
        KeyType::Binary => var_odd_binary_poly_key_square(poly_size),
        KeyType::Ternary => var_odd_ternary_poly_key_square(poly_size),
        KeyType::Gaussian => var_odd_gaussian_poly_key_square(poly_size),
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the even coefficients of a polynomial key to the square 
/// given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The noise variance of the even coefficients of a polynomial key to the square 
pub fn var_even_poly_key_square(poly_size: usize, key_type: KeyType) -> f64 {
    if poly_size == 1 {
        return 2. * var_coef_key_square(key_type);
    }
    match key_type {
        KeyType::Binary => var_even_binary_poly_key_square(poly_size),
        KeyType::Ternary => var_even_ternary_poly_key_square(poly_size),
        KeyType::Gaussian => var_even_gaussian_poly_key_square(poly_size),
        KeyType::Zero => 0.,
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key to the square 
/// given the key type
/// Arguments:
/// * `key_type` - input key type
/// Output
/// * The mean expectation of the coefficients of a polynomial key to the square
pub fn square_expect_mean_poly_key_square(poly_size: usize, key_type: KeyType) -> f64 {
    if poly_size == 1 {
        return square(expect_coef_key_square(key_type));
    }
    match key_type {
        KeyType::Binary => square_expect_mean_binary_poly_key_square(poly_size),
        KeyType::Ternary => square_expect_mean_ternary_poly_key_square(poly_size),
        KeyType::Gaussian => square_expect_mean_gaussian_poly_key_square(poly_size),
        KeyType::Zero => 0.,
    }
}

/// Returns the variance of the coefficients of a polynomial key resulting from 
/// the multiplication of two polynomial keys of the same type (S_i x S_j)
/// given their key type
/// Arguments:
/// * `key_type` - input key type (both keys in the product have the same key type)
/// Output
/// * The noise variance of the coefficients of the polynomial key S_i x S_j 
pub fn var_poly_key_times_key(poly_size: usize, key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => var_binary_poly_key_times_key(poly_size),
        KeyType::Ternary => var_ternary_poly_key_times_key(poly_size),
        KeyType::Gaussian => var_gaussian_poly_key_times_key(poly_size),
        KeyType:Zero => 0.,
    }
}

/// Returns the mean expectation of the coefficients of a polynomial key resulting from 
/// the multiplication of two polynomial keys of the same type (S_i x S_j)
/// given their key type
/// Arguments:
/// * `key_type` - input key type (both keys in the product have the same key type)
/// Output
/// * The mean expectation of the coefficients of the polynomial key S_i x S_j 
pub fn square_expect_mean_poly_key_times_key(poly_size: usize, key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => square_expect_mean_binary_poly_key_times_key(poly_size),
        KeyType::Ternary => square_expect_mean_ternary_poly_key_times_key(poly_size),
        KeyType::Gaussian => square_expect_mean_gaussian_poly_key_times_key(poly_size),
        KeyType::Zero => 0.,
    }
}

/// Binary Keys: coefficients

/// Returns the variance of a coefficient of a binary secret key
fn var_binary_key_coefficient() -> f64 {
    1. / 4.
}

/// Returns the expectation of a coefficient of a binary secret key
fn expect_binary_key_coefficient() -> f64 {
    1. / 2.
}

/// Binary Keys: squared coefficients

/// Returns the variance of a squared binary coefficient
fn var_binary_coef_key_square() -> f64 {
    1. / 4.
}

/// Returns the expectation of a squared binary coefficient
fn expect_binary_coef_key_square() -> f64 {
    1. / 2.
}

/// Binary Keys: polynomial coefficients of S_i x S_j, named: 
/// * Squared Binary Key - if i=j
/// * Product of 2 different Binary Keys - if i!=j

/// Returns the variance of the odd coefficients in a squared binary key
fn var_odd_binary_poly_key_square(poly_size: usize) -> f64 {
    3. * (poly_size as f64) / 8.
}

/// Returns the variance of the even coefficients in a squared binary key
fn var_even_binary_poly_key_square(poly_size: usize) -> f64 {
    ((3 * poly_size + 2) as f64) / 16.
}

/// Returns the mean of the square expectation of the coefficients in a squared binary key
fn square_expect_mean_binary_poly_key_square(poly_size: usize) -> f64 {
    (square(poly_size as f64) + 2.) / 48.
}

/// Returns the variance of the coefficients in a product of 2 different binary keys
fn var_binary_poly_key_times_key(poly_size: usize) -> f64 {
    3. * (poly_size as f64) / 16.
}

/// Returns the mean of the square expectation of the coefficients in a product of 2 different binary keys
fn square_expect_mean_binary_poly_key_times_key(poly_size: usize) -> f64 {
    (square(poly_size as f64) + 2.) / 48.
}


/// Ternary Keys: coefficients

/// Returns the variance of a coefficient of a ternary secret key
fn var_ternary_key_coefficient() -> f64 {
    2. / 3.
}

/// Returns the expectation of a coefficient of a ternary secret key
fn expect_ternary_key_coefficient() -> f64 {
    0.
}

/// Ternary Keys: squared coefficients

/// Returns the variance of a squared ternary coefficient
fn var_ternary_coef_key_square() -> f64 {
    2. / 9.
}

/// Returns the expectation of a squared ternary coefficient
fn expect_ternary_coef_key_square() -> f64 {
    2. / 3.
}

/// Ternary Keys: polynomial coefficients of S_i x S_j, named: 
/// * Squared Ternary Key - if i=j
/// * Product of 2 different Ternary Keys - if i!=j

/// Returns the variance of the odd coefficients in a ternary key squared
fn var_odd_ternary_poly_key_square(poly_size: usize) -> f64 {
    8. * (poly_size as f64) / 9.
}

/// Returns the variance of the even coefficients in a ternary key squared
fn var_even_ternary_poly_key_square(poly_size: usize) -> f64 {
    4. * ((2 * poly_size - 3) as f64) / 9.
}

/// Returns the mean of the square expectation of the coefficients in a ternary key squared
fn square_expect_mean_ternary_poly_key_square(_poly_size: usize) -> f64 {
    0.
}

/// Returns the variance of the coefficients in a product of 2 different ternary keys
fn var_ternary_poly_key_times_key(poly_size: usize) -> f64 {
    4. * (poly_size as f64) / 9.
}

/// Returns the mean of the square expectation of the coefficients in a product of 2 different binary keys
fn square_expect_mean_ternary_poly_key_times_key(_poly_size: usize) -> f64 {
    0.
}

/// Gaussian Keys: coefficients

/// Returns the variance of a coefficient of a gaussian secret key
fn var_gaussian_key_coefficient() -> f64 {
    square(GAUSSIAN_STDEV * f64::powi(2., 64 as i32))
}

/// Returns the expectation of a coefficient of a gaussian secret key
fn expect_gaussian_key_coefficient() -> f64 {
    0.
}

/// Gaussian Keys: squared coefficients

/// Returns the variance of a squared Gaussian coefficient
fn var_gaussian_coef_key_square() -> f64 {
    2. * square(var_gaussian_key_coefficient())
}

/// Returns the expectation of a squared Gaussian coefficient
fn expect_gaussian_coef_key_square() -> f64 {
    var_gaussian_key_coefficient()
}

/// Gaussian Keys: polynomial coefficients of S_i x S_j, named: 
/// * Squared Gaussian Key - if i=j
/// * Product of 2 different Gaussian Keys - if i!=j

/// Returns the variance of the odd coefficients in a gaussian key squared
fn var_odd_gaussian_poly_key_square(poly_size: usize) -> f64 {
    2. * (poly_size as f64) * square(var_gaussian_key_coefficient())
}

/// Returns the variance of the even coefficients in a gaussian key squared
fn var_even_gaussian_poly_key_square(poly_size: usize) -> f64 {
    var_odd_gaussian_poly_key_square(poly_size)
}

/// Returns the mean of the square expectation of the coefficients in a gaussian key squared
fn square_expect_mean_gaussian_poly_key_square(_poly_size: usize) -> f64 {
    0.
}

/// Returns the variance of the coefficients in a product of 2 different gaussian keys
fn var_gaussian_poly_key_times_key(poly_size: usize) -> f64 {
    square(var_gaussian_key_coefficient()) * (poly_size as f64)
}

/// Returns the mean of the square expectation of the coefficients in a product of 2 different gaussian keys
fn square_expect_mean_gaussian_poly_key_times_key(_poly_size: usize) -> f64 {
    0.
}

/// Square function tool
/// Arguments:
/// * `x` - input 
/// Output
/// * x^2 
pub fn square(x: f64) -> f64 {
    x * x
}
