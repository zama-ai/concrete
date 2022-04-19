use concrete_commons::dispersion::Variance;

use crate::parameters::GlweParameters;

/// Noise ensuring security
// It was 128 bits of security on the 30th August 2021 with https://bitbucket.org/malb/lwe-estimator/commits/fb7deba98e599df10b665eeb6a26332e43fb5004
pub fn minimal_variance(
    glwe_params: GlweParameters,
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/security.py
    // ensure to have a minimal on std deviation covering the 2 lowest bits on modular scale
    assert!(
        security_level == 128,
        "Only 128 bits of security is supported"
    );
    let espsilon_log2_std_modular = 2.0;
    let espsilon_log2_std = espsilon_log2_std_modular - (ciphertext_modulus_log as f64);
    let equiv_lwe_dimension = (glwe_params.glwe_dimension * glwe_params.polynomial_size()) as f64;
    let secure_log2_std = -0.026374888765705498 * equiv_lwe_dimension + 2.012143923330495;
    // TODO: could be added instead
    let log2_std = f64::max(secure_log2_std, espsilon_log2_std);
    let log2_var = 2.0 * log2_std;
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
        let golden_std_dev = 2.168404344971009e-19;
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
        let golden_std_dev = 3.2216458741669603e-6;
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
