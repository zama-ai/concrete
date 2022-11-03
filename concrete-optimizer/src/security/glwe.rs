use super::security_weights::security_weight;
use crate::parameters::GlweParameters;
use concrete_commons::dispersion::Variance;

/// Noise ensuring security
pub fn minimal_variance(
    glwe_params: GlweParameters,
    ciphertext_modulus_log: u32,
    security_level: u64,
) -> Variance {
    let equiv_lwe_dimension = glwe_params.glwe_dimension * glwe_params.polynomial_size();
    let security_weights = security_weight(security_level)
        .unwrap_or_else(|| panic!("{security_level} bits of security is not supported"));

    let secure_log2_std =
        security_weights.secure_log2_std(equiv_lwe_dimension, ciphertext_modulus_log as f64);
    let log2_var = 2.0 * secure_log2_std;
    Variance(f64::exp2(log2_var))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concrete_commons::dispersion::DispersionParameter;

    #[test]
    fn golden_python_prototype_security_security_glwe_variance_low() {
        // python securityFunc(10,14,64)= 0.3120089883926036
        let integer_size = 64;
        let golden_std_dev = 2.168_404_344_971_009e-19;
        let security_level = 128;
        let glwe_params = GlweParameters {
            log2_polynomial_size: 14,
            glwe_dimension: 10,
        };
        let actual = minimal_variance(glwe_params, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            actual.get_standard_dev(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn golden_python_prototype_security_security_glwe_variance_high() {
        // python securityFunc(3,8,32)= 2.6011445832514504
        let integer_size = 32;
        let golden_std_dev = 4.392_824_146_816_922_4e-6;
        let security_level = 128;
        let glwe_params = GlweParameters {
            log2_polynomial_size: 8,
            glwe_dimension: 3,
        };

        let actual = minimal_variance(glwe_params, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            actual.get_standard_dev(),
            epsilon = f64::EPSILON
        );
    }
}
