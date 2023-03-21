mod security_weight;

pub use security_weight::{security_weight, supported_security_levels};

/// Noise ensuring security
pub fn minimal_variance_lwe(
    lwe_dimension: u64,
    ciphertext_modulus_log: u32,
    security_level: u64,
) -> f64 {
    minimal_variance_glwe(lwe_dimension, 1, ciphertext_modulus_log, security_level)
}

/// Noise ensuring security
pub fn minimal_variance_glwe(
    glwe_dimension: u64,
    polynomial_size: u64,
    ciphertext_modulus_log: u32,
    security_level: u64,
) -> f64 {
    let equiv_lwe_dimension = glwe_dimension * polynomial_size;
    let security_weights = security_weight(security_level)
        .unwrap_or_else(|| panic!("{security_level} bits of security is not supported"));

    let secure_log2_std =
        security_weights.secure_log2_std(equiv_lwe_dimension, ciphertext_modulus_log as f64);
    let log2_var = 2.0 * secure_log2_std;
    f64::exp2(log2_var)
}

#[cfg(test)]
mod tests {
    use super::{super::conversion::variance_to_std_dev, minimal_variance_glwe};

    #[test]
    fn golden_python_prototype_security_security_glwe_variance_low() {
        // python securityFunc(10,14,64)= 0.3120089883926036
        let integer_size = 64;
        let golden_std_dev = 2.168_404_344_971_009e-19;
        let security_level = 128;

        let actual = minimal_variance_glwe(10, 1 << 14, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            variance_to_std_dev(actual),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn golden_python_prototype_security_security_glwe_variance_high() {
        // python securityFunc(3,8,32)= 2.6011445832514504
        let integer_size = 32;
        let golden_std_dev = 4.392_824_146_816_922_4e-6;
        let security_level = 128;

        let actual = minimal_variance_glwe(3, 1 << 8, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            variance_to_std_dev(actual),
            epsilon = f64::EPSILON
        );
    }
}
