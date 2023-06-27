use crate::{gaussian_noise::conversion::modular_variance_to_variance, utils::square};

pub fn variance_keyswitch_glwe(
    input_glwe_dimension: u64,
    variance_input_glwe: f64,
    polynomial_size: u64,
    log2_base: u64,
    level: u64,
    ciphertext_modulus_log: u32,
    variance_ksk: f64,
) -> f64 {
    let k = input_glwe_dimension as f64;
    let b_square = 2_f64.powi(2 * log2_base as i32);
    let b2l = 2_f64.powi((log2_base * 2 * level) as i32);
    let l = level as f64;
    let big_n = polynomial_size as f64;
    let q_square = 2_f64.powi(ciphertext_modulus_log as i32);
    let variance_key_coefficient_binary: f64 =
        modular_variance_to_variance(1. / 4., ciphertext_modulus_log);
    let square_expectation_key_coefficient_binary: f64 =
        modular_variance_to_variance(square(1. / 2.), ciphertext_modulus_log);

    variance_input_glwe
        + k * big_n
            * ((q_square - b2l) / (12. * b2l))
            * (variance_key_coefficient_binary + square_expectation_key_coefficient_binary)
        + k * big_n / 4. * variance_key_coefficient_binary
        + k * big_n * l * variance_ksk * (b_square + 2.) / 12.
}
