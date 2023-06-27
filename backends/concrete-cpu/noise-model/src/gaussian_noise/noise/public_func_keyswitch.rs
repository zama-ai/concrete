use crate::utils::square;

pub fn variance_public_functional_keyswitch(
    variance_input_lwe: f64,
    lipschitz_constant: f64,
    lwe_dimension_input: u64,
    ciphertext_modulus_log: i32,
    log2_base: u64,
    level: u64,
    variance_ksk: f64,
) -> f64 {
    let variance_key_coefficient_binary: f64 = 1. / 4.;
    let expectation_key_coefficient_binary: f64 = 1. / 2.;
    let lwe_dimension = lwe_dimension_input as f64;

    let l = level as f64;
    let b = 2f64.powi(log2_base as i32);
    let b2l = f64::powi(b, 2 * level as i32);
    let q_square = 2_f64.powi(2 * ciphertext_modulus_log);

    let res_1 = square(lipschitz_constant * variance_input_lwe);

    let res_2 = lwe_dimension * (q_square - b2l) / (12. * b2l)
        * (variance_key_coefficient_binary + square(expectation_key_coefficient_binary))
        + lwe_dimension / 4. * variance_key_coefficient_binary;

    let res_3 = lwe_dimension * l * variance_ksk * (square(b) + 2.) / 12.;

    res_1 + res_2 + res_3
}
