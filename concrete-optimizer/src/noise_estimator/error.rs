use concrete_commons::dispersion::{DispersionParameter, Variance};

use crate::utils::square;

pub fn sigma_scale_of_error_probability(p_error: f64) -> f64 {
    // https://en.wikipedia.org/wiki/Error_function#Applications
    statrs::function::erf::erfc_inv(p_error) * 2_f64.sqrt()
}

pub fn error_probability_of_sigma_scale(sigma_scale: f64) -> f64 {
    statrs::function::erf::erfc(sigma_scale / 2_f64.sqrt())
}

const LEFT_PADDING_BITS: u64 = 1;
const RIGHT_PADDING_BITS: u64 = 1;

pub fn fatal_variance_limit(padding_bits: u64, precision: u64, ciphertext_modulus_log: u32) -> f64 {
    let no_noise_bits = padding_bits + precision;
    let noise_bits = ciphertext_modulus_log as u64 - no_noise_bits;
    2_f64.powi(noise_bits as i32)
}

fn safe_variance_bound_from_p_error(
    fatal_noise_limit: f64,
    ciphertext_modulus_log: u32,
    maximum_acceptable_error_probability: f64,
) -> f64 {
    // We want safe_sigma such that:
    // P(x not in [-+fatal_noise_limit] | σ = safe_sigma) = p_error
    // <=> P(x not in [-+fatal_noise_limit/safe_sigma] | σ = 1) = p_error
    // <=> P(x not in [-+kappa] | σ = 1) = p_error, with safe_sigma = fatal_noise_limit / kappa
    let kappa = sigma_scale_of_error_probability(maximum_acceptable_error_probability);
    let safe_sigma = fatal_noise_limit / kappa;
    let modular_variance = square(safe_sigma);

    Variance::from_modular_variance(modular_variance, ciphertext_modulus_log).get_variance()
}

pub fn safe_variance_bound_2padbits(
    precision: u64,
    ciphertext_modulus_log: u32,
    maximum_acceptable_error_probability: f64,
) -> f64 {
    let padding_bits = LEFT_PADDING_BITS + RIGHT_PADDING_BITS;
    let fatal_noise_limit = fatal_variance_limit(padding_bits, precision, ciphertext_modulus_log);
    safe_variance_bound_from_p_error(
        fatal_noise_limit,
        ciphertext_modulus_log,
        maximum_acceptable_error_probability,
    )
}

pub fn safe_variance_bound_1bit_1padbit(
    ciphertext_modulus_log: u32,
    maximum_acceptable_error_probability: f64,
) -> f64 {
    // This is hardcoded and only valid for 16bit wop pbs
    // Precision is 1
    let noise_bits = 58;
    let fatal_noise_limit = 2_f64.powi(noise_bits as i32);
    safe_variance_bound_from_p_error(
        fatal_noise_limit,
        ciphertext_modulus_log,
        maximum_acceptable_error_probability,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmas() {
        // https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_coverage
        let reference = &[
            0.682_689_492_137, // +- 1 sigma
            0.954_499_736_104, // 2
            0.997_300_203_937, // ...
            0.999_936_657_516,
            0.999_999_426_697,
        ];
        for (i, &p_in) in reference.iter().enumerate() {
            let p_out = 1.0 - p_in;
            let expected_scale = (i + 1) as f64;
            approx::assert_relative_eq!(
                expected_scale,
                sigma_scale_of_error_probability(p_out),
                max_relative = 1e-8
            );
            approx::assert_relative_eq!(
                p_out,
                error_probability_of_sigma_scale(sigma_scale_of_error_probability(p_out)),
                max_relative = 1e-8
            );
        }
    }
}
