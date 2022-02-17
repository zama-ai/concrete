use concrete_commons::dispersion::Variance;

/// Noise ensuring security
// It was 128 bits of security on the 30th August 2021 with https://bitbucket.org/malb/lwe-estimator/commits/fb7deba98e599df10b665eeb6a26332e43fb5004
pub fn variance_glwe(
    glwe_polynomial_size: u64,
    glwe_dimension: u64,
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/security.py
    // ensure to have a minimal on std deviation covering the 2 lowest bits on modular scale
    if security_level != 128 {
        panic!("Only 128 bits of security is supported")
    }
    let espsilon_log2_std_modular = 2.0;
    let espsilon_log2_std = espsilon_log2_std_modular - (ciphertext_modulus_log as f64);
    let equiv_lwe_dimension = (glwe_dimension * glwe_polynomial_size) as f64;
    let secure_log2_std = -0.026374888765705498 * equiv_lwe_dimension + 2.012143923330495;
    // TODO: could be added instead
    let log2_std = f64::max(secure_log2_std, espsilon_log2_std);
    let log2_var = 2.0 * log2_std;
    Variance(f64::exp2(log2_var))
}

/// Noise ensuring ksk security
pub fn variance_ksk(
    glwe_polynomial_size: u64,
    glwe_dimension: u64,
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L13
    variance_glwe(
        glwe_polynomial_size,
        glwe_dimension,
        ciphertext_modulus_log,
        security_level,
    )
}

/// Noise ensuring bsk security
pub fn variance_bsk(
    glwe_polynomial_size: u64,
    glwe_dimension: u64,
    ciphertext_modulus_log: u64,
    security_level: u64,
) -> Variance {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L66
    variance_glwe(
        glwe_polynomial_size,
        glwe_dimension,
        ciphertext_modulus_log,
        security_level,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use concrete_commons::dispersion::DispersionParameter;

    #[test]
    fn golden_python_prototype_security_variance_glwe_low() {
        // python securityFunc(10,14,64)= 0.3120089883926036
        let log_poly_size = 14;
        let glwe_dimension = 10;
        let integer_size = 64;
        let golden_std_dev = 0.312_008_988_392_6036;
        let security_level = 128;
        let actual = variance_glwe(log_poly_size, glwe_dimension, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            actual.get_standard_dev(),
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn golden_python_prototype_security_variance_glwe_high() {
        // python securityFunc(3,8,32)= 2.6011445832514504
        let log_poly_size = 8;
        let glwe_dimension = 3;
        let integer_size = 32;
        let golden_std_dev = 2.6011445832514504;
        let security_level = 128;
        let actual = variance_glwe(log_poly_size, glwe_dimension, integer_size, security_level);
        approx::assert_relative_eq!(
            golden_std_dev,
            actual.get_standard_dev(),
            epsilon = f64::EPSILON
        );
    }
}
