pub enum KeyType {
    Binary,
    Ternary,
    Gaussian,
    Zero,
}

pub fn var_key_coefficient(key_type: KeyType) -> f64 {
    match key_type {
        KeyType::Binary => var_binary_key_coefficient(),
        KeyType::Ternary => var_ternary_key_coefficient(),
        KeyType::Gaussian => var_gaussian_key_coefficient(),
        KeyType::Zero => 0.,
    }
}

pub fn expect_key_coefficient(key_type: char) -> f64 {
    match key_type {
        BINARY_KEY => expect_binary_key_coefficient(),
        TERNARY_KEY => expect_ternary_key_coefficient(),
        GAUSSIAN_KEY => expect_gaussian_key_coefficient(),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

pub fn var_coef_key_square(key_type: char) -> f64 {
    match key_type {
        BINARY_KEY => var_binary_coef_key_square(),
        TERNARY_KEY => var_ternary_coef_key_square(),
        GAUSSIAN_KEY => var_gaussian_coef_key_square(),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

pub fn expect_coef_key_square(key_type: char) -> f64 {
    match key_type {
        BINARY_KEY => expect_binary_coef_key_square(),
        TERNARY_KEY => expect_ternary_coef_key_square(),
        GAUSSIAN_KEY => expect_gaussian_coef_key_square(),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

pub fn var_odd_poly_key_square(poly_size: usize, key_type: char) -> f64 {
    if poly_size == 1 {
        return 0.;
    }
    match key_type {
        BINARY_KEY => var_odd_binary_poly_key_square(poly_size),
        TERNARY_KEY => var_odd_ternary_poly_key_square(poly_size),
        GAUSSIAN_KEY => var_odd_gaussian_poly_key_square(poly_size),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

pub fn var_even_poly_key_square(poly_size: usize, key_type: char) -> f64 {
    if poly_size == 1 {
        return 2. * var_coef_key_square(key_type);
    }
    match key_type {
        BINARY_KEY => var_even_binary_poly_key_square(poly_size),
        TERNARY_KEY => var_even_ternary_poly_key_square(poly_size),
        GAUSSIAN_KEY => var_even_gaussian_poly_key_square(poly_size),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

pub fn square_expect_mean_poly_key_square(poly_size: usize, key_type: char) -> f64 {
    if poly_size == 1 {
        return square(expect_coef_key_square(key_type));
    }
    match key_type {
        BINARY_KEY => square_expect_mean_binary_poly_key_square(poly_size),
        TERNARY_KEY => square_expect_mean_ternary_poly_key_square(poly_size),
        GAUSSIAN_KEY => square_expect_mean_gaussian_poly_key_square(poly_size),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

// S_i \times S_j
pub fn var_poly_key_times_key(poly_size: usize, key_type: char) -> f64 {
    match key_type {
        BINARY_KEY => var_binary_poly_key_times_key(poly_size),
        TERNARY_KEY => var_ternary_poly_key_times_key(poly_size),
        GAUSSIAN_KEY => var_gaussian_poly_key_times_key(poly_size),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

// S_i \times S_j
pub fn square_expect_mean_poly_key_times_key(poly_size: usize, key_type: char) -> f64 {
    match key_type {
        BINARY_KEY => square_expect_mean_binary_poly_key_times_key(poly_size),
        TERNARY_KEY => square_expect_mean_ternary_poly_key_times_key(poly_size),
        GAUSSIAN_KEY => square_expect_mean_gaussian_poly_key_times_key(poly_size),
        ZERO_KEY => 0.,
        _ => panic!("wrong key type"),
    }
}

// Binary Keys

/// gives the variance of a coefficient of a binary secret key
fn var_binary_key_coefficient() -> f64 {
    1. / 4.
}

/// gives the expectation of a coefficient of a binary secret key
fn expect_binary_key_coefficient() -> f64 {
    1. / 2.
}

// square coefficients

/// gives the variance of a squared binary coefficient
fn var_binary_coef_key_square() -> f64 {
    1. / 4.
}

/// gives the expectation of a squared binary coefficient
fn expect_binary_coef_key_square() -> f64 {
    1. / 2.
}

// Squared Binary Polynomial Keys

/// compute the variance of the odd coefficients in a binary key squared
fn var_odd_binary_poly_key_square(poly_size: usize) -> f64 {
    3. * (poly_size as f64) / 8.
}

/// compute the variance of the even coefficients in a binary key squared
fn var_even_binary_poly_key_square(poly_size: usize) -> f64 {
    ((3 * poly_size + 2) as f64) / 16.
}

/// compute the mean of the square expectation of the coefficients in a binary key squared
fn square_expect_mean_binary_poly_key_square(poly_size: usize) -> f64 {
    (square(poly_size as f64) + 2.) / 48.
}

/// compute the variance of the coefficients in a product of 2 different binary keys
fn var_binary_poly_key_times_key(poly_size: usize) -> f64 {
    3. * (poly_size as f64) / 16.
}

/// compute the mean of the square expectation of the coefficients in a product of 2 different binary keys
fn square_expect_mean_binary_poly_key_times_key(poly_size: usize) -> f64 {
    (square(poly_size as f64) + 2.) / 48.
}

// Ternary Keys

/// gives the variance of a coefficient of a ternary secret key
fn var_ternary_key_coefficient() -> f64 {
    2. / 3.
}

/// gives the expectation of a coefficient of a ternary secret key
fn expect_ternary_key_coefficient() -> f64 {
    0.
}

// square coefficients

/// gives the variance of a squared ternary coefficient
fn var_ternary_coef_key_square() -> f64 {
    2. / 9.
}

/// gives the expectation of a squared ternary coefficient
fn expect_ternary_coef_key_square() -> f64 {
    2. / 3.
}

// Squared Polynomial Ternary Keys

/// compute the variance of the odd coefficients in a ternary key squared
fn var_odd_ternary_poly_key_square(poly_size: usize) -> f64 {
    8. * (poly_size as f64) / 9.
}

/// compute the variance of the even coefficients in a ternary key squared
fn var_even_ternary_poly_key_square(poly_size: usize) -> f64 {
    4. * ((2 * poly_size - 3) as f64) / 9.
}

/// compute the mean of the square expectation of the coefficients in a ternary key squared
fn square_expect_mean_ternary_poly_key_square(_poly_size: usize) -> f64 {
    0.
}

/// compute the variance of the coefficients in a product of 2 different ternary keys
fn var_ternary_poly_key_times_key(poly_size: usize) -> f64 {
    4. * (poly_size as f64) / 9.
}

/// compute the mean of the square expectation of the coefficients in a product of 2 different binary keys
fn square_expect_mean_ternary_poly_key_times_key(_poly_size: usize) -> f64 {
    0.
}

// Gaussian Keys

/// gives the variance of a coefficient of a ternary secret key
fn var_gaussian_key_coefficient() -> f64 {
    square(GAUSSIAN_STDEV * f64::powi(2., 64 as i32))
}

/// gives the expectation of a coefficient of a ternary secret key
fn expect_gaussian_key_coefficient() -> f64 {
    0.
}

// square coefficients

/// gives the variance of a squared Gaussian coefficient
fn var_gaussian_coef_key_square() -> f64 {
    2. * square(var_gaussian_key_coefficient())
}

/// gives the expectation of a squared Gaussian coefficient
fn expect_gaussian_coef_key_square() -> f64 {
    var_gaussian_key_coefficient()
}

// Squared Gaussian Polynomial Keys

/// compute the variance of the odd coefficients in a ternary key squared
fn var_odd_gaussian_poly_key_square(poly_size: usize) -> f64 {
    2. * (poly_size as f64) * square(var_gaussian_key_coefficient())
}

/// compute the variance of the even coefficients in a ternary key squared
fn var_even_gaussian_poly_key_square(poly_size: usize) -> f64 {
    var_odd_gaussian_poly_key_square(poly_size)
}

/// compute the mean of the square expectation of the coefficients in a gaussian key squared
fn square_expect_mean_gaussian_poly_key_square(_poly_size: usize) -> f64 {
    0.
}

/// compute the variance of the coefficients in a product of 2 different gaussian keys
fn var_gaussian_poly_key_times_key(poly_size: usize) -> f64 {
    square(var_gaussian_key_coefficient()) * (poly_size as f64)
}

/// compute the mean of the square expectation of the coefficients in a product of 2 different binary keys
fn square_expect_mean_gaussian_poly_key_times_key(_poly_size: usize) -> f64 {
    0.
}

// tool

pub fn square(x: f64) -> f64 {
    x * x
}
