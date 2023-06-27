use crate::{gaussian_noise::conversion::modular_variance_to_variance, utils::square};
use std::cmp;

pub fn variance_tensor_product_glwe(
    glwe_dimension: u64,
    polynomial_size: u64,
    ciphertext_modulus_log: u32,
    variance_glwe1: f64,
    variance_glwe2: f64,
    scale1: u64,
    scale2: u64,
    two_norm_m1: u64,
    two_norm_m2: u64,
) -> f64 {
    //use formula's from CLOT21 appendix C.1
    let scale = cmp::min(scale1, scale2) as f64;
    let sq_scale = scale.powi(2) as f64;
    let poly_size = polynomial_size as f64;
    let glwe_dim = glwe_dimension as f64;
    let norm_m1 = two_norm_m1 as f64;
    let norm_m2 = two_norm_m2 as f64;
    let var_first = (poly_size / sq_scale)
        * ((scale1 as f64).powi(2) * norm_m1.powi(2) * variance_glwe2
            + (scale2 as f64).powi(2) * norm_m2.powi(2) * variance_glwe1
            + variance_glwe1 * variance_glwe2);
    let q_square = 2_f64.powi(2 * ciphertext_modulus_log as i32);
    let variance_key_coefficient_binary: f64 =
        modular_variance_to_variance(1. / 4., ciphertext_modulus_log);

    let square_expectation_key_coefficient_binary: f64 =
        modular_variance_to_variance(square(1. / 2.), ciphertext_modulus_log);

    //for the keys use the defined values??
    let var_second = (poly_size / sq_scale)
        * ((q_square - 1.) / 12.
            * (1.
                + glwe_dim * poly_size * variance_key_coefficient_binary
                + glwe_dim * poly_size * square_expectation_key_coefficient_binary)
            + ((glwe_dim * poly_size) / 4.) * variance_key_coefficient_binary
            + 1. / 4. * (1. + glwe_dim * poly_size * square_expectation_key_coefficient_binary))
        * (variance_glwe1 + variance_glwe2);

    let variance_keyprime_odd_coefficient_binary;
    let variance_keyprime_even_coefficient_binary;
    let square_expectation_keyprime_coefficient_binary;
    if polynomial_size == 1 {
        variance_keyprime_odd_coefficient_binary = 0.;
        variance_keyprime_even_coefficient_binary = 1. / 2.;
        square_expectation_keyprime_coefficient_binary = 1. / 2.;
    } else {
        variance_keyprime_odd_coefficient_binary = 3. / 8. * poly_size;
        variance_keyprime_even_coefficient_binary = 3. / 8. * poly_size - 1. / 4.;
        square_expectation_keyprime_coefficient_binary = (poly_size.powi(2) + 2.) / 48.;
    }
    let variance_keyprimeprime_coefficient_binary = 3. / 16. * poly_size;
    let square_expectation_keyprimeprime_coefficient_binary = (poly_size.powi(2) + 2.) / 48.;

    let var_third = 1. / 12.
        + (glwe_dim * poly_size) / (12. * sq_scale)
            * ((sq_scale - 1.)
                * (variance_key_coefficient_binary + square_expectation_key_coefficient_binary)
                + 3. * variance_key_coefficient_binary)
        + (glwe_dim * (glwe_dim - 1.) * poly_size) / (24. * sq_scale)
            * ((sq_scale - 1.)
                * (variance_keyprimeprime_coefficient_binary
                    + square_expectation_keyprimeprime_coefficient_binary)
                + 3. * variance_keyprimeprime_coefficient_binary)
        + (glwe_dim * poly_size) / (24. * sq_scale)
            * ((sq_scale - 1.)
                * (variance_keyprime_odd_coefficient_binary
                    + variance_keyprime_even_coefficient_binary
                    + 2. * square_expectation_keyprime_coefficient_binary)
                + 3. * (variance_keyprime_odd_coefficient_binary
                    + variance_keyprime_even_coefficient_binary));

    var_first + var_second + var_third
}

pub fn variance_glwe_relin(
    variance_input_glwe: f64,
    glwe_dimension: u64,
    polynomial_size: u64,
    ciphertext_modulus_log: u32,
    log2_base: u64,
    level: u64,
    variance_rlk: f64,
) -> f64 {
    let b = 2_f64.powi(log2_base as i32);
    let l = level as f64;
    let poly_size = polynomial_size as f64;
    let glwe_dim = glwe_dimension as f64;
    let var_part_one = variance_input_glwe
        + glwe_dim * l * poly_size * variance_rlk * (glwe_dim + 1.) / 2. * (b.powi(2) + 2.) / 12.;

    let q_square = 2_f64.powi(2 * ciphertext_modulus_log as i32);
    let b2l = 2_f64.powi((log2_base * 2 * level) as i32);

    let variance_keyprime_odd_coefficient_binary;
    let variance_keyprime_even_coefficient_binary;
    let square_expectation_keyprime_coefficient_binary;
    if polynomial_size == 1 {
        variance_keyprime_odd_coefficient_binary = 0.;
        variance_keyprime_even_coefficient_binary = 1. / 2.;
        square_expectation_keyprime_coefficient_binary = 1. / 2.;
    } else {
        variance_keyprime_odd_coefficient_binary = 3. / 8. * poly_size;
        variance_keyprime_even_coefficient_binary = 3. / 8. * poly_size - 1. / 4.;
        square_expectation_keyprime_coefficient_binary = (poly_size.powi(2) + 2.) / 48.;
    }
    let variance_keyprimeprime_coefficient_binary = 3. / 16. * poly_size;
    let square_expectation_keyprimeprime_coefficient_binary = (poly_size.powi(2) + 2.) / 48.;
    let var_part_two = (glwe_dim * poly_size) / 2.
        * ((q_square / (12. * b2l)) - 1. / 12.)
        * ((glwe_dim - 1.)
            * (variance_keyprimeprime_coefficient_binary
                + square_expectation_keyprimeprime_coefficient_binary)
            + variance_keyprime_odd_coefficient_binary
            + variance_keyprime_even_coefficient_binary
            + 2. * square_expectation_keyprime_coefficient_binary)
        + (glwe_dim * poly_size) / 8.
            * ((glwe_dim - 1.) * variance_keyprimeprime_coefficient_binary
                + variance_keyprime_odd_coefficient_binary
                + variance_keyprime_even_coefficient_binary);

    var_part_one + var_part_two
}

pub fn variance_tensor_product_with_glwe_relin(
    glwe_dimension: u64,
    polynomial_size: u64,
    ciphertext_modulus_log: u32,
    variance_glwe1: f64,
    variance_glwe2: f64,
    scale1: u64,
    scale2: u64,
    two_norm_m1: u64,
    two_norm_m2: u64,
    log2_base: u64,
    level: u64,
    variance_rlk: f64,
) -> f64 {
    variance_glwe_relin(
        variance_tensor_product_glwe(
            glwe_dimension,
            polynomial_size,
            ciphertext_modulus_log,
            variance_glwe1,
            variance_glwe2,
            scale1,
            scale2,
            two_norm_m1,
            two_norm_m2,
        ),
        glwe_dimension,
        polynomial_size,
        ciphertext_modulus_log,
        log2_base,
        level,
        variance_rlk,
    )
}
