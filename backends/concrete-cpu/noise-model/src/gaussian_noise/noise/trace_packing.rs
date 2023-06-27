use crate::{gaussian_noise::conversion::modular_variance_to_variance, utils::square};

pub fn variance_trace_packing_keyswitch(
    variance_input_lwe: f64,
    glwe_dimension: u64,
    polynomial_size: u64,
    log2_base: u64,
    level: u64,
    ciphertext_modulus_log: u32,
    variance_ksk: f64,
) -> f64 {
    glwe_dimension as f64
        * (polynomial_size as f64).log2()
        * variance_coefficient_glwe_keyswitch(
            variance_input_lwe,
            glwe_dimension,
            polynomial_size,
            ciphertext_modulus_log,
            log2_base,
            level,
            variance_ksk,
        )
}

/// Noise of each coefficient of the output GLWE
pub fn variance_coefficient_glwe_keyswitch(
    variance_input_lwe: f64,
    glwe_dimension: u64,
    polynomial_size: u64,
    ciphertext_modulus_log: u32,
    log2_base: u64,
    level: u64,
    variance_ksk: f64,
) -> f64 {
    let variance_key_coefficient_binary: f64 =
        modular_variance_to_variance(1. / 4., ciphertext_modulus_log);

    let square_expectation_key_coefficient_binary: f64 =
        modular_variance_to_variance(square(1. / 2.), ciphertext_modulus_log);

    let base = 2_f64.powi(log2_base as i32);

    let b2l = 2_f64.powi((log2_base * 2 * level) as i32);
    let q_square = 2_f64.powi((2 * ciphertext_modulus_log) as i32);
    let l = level as f64;
    let poly_size = polynomial_size as f64;
    let glwe_dim = glwe_dimension as f64;

    variance_input_lwe
        + glwe_dim * poly_size * (q_square - b2l) / (12. * b2l)
            * (variance_key_coefficient_binary + square_expectation_key_coefficient_binary)
        + glwe_dim * poly_size / 4. * variance_key_coefficient_binary
        + l * glwe_dim * poly_size * variance_ksk * (base.powi(2) + 2.) / 12.
}
