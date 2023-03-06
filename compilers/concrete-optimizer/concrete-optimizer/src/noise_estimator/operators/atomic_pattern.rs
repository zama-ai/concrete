use crate::parameters::AtomicPatternParameters;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_security_curves::gaussian::security::minimal_variance_lwe;

pub fn maximal_noise(
    input_variance: f64,
    param: AtomicPatternParameters,
    ciphertext_modulus_log: u32, //log(q)
    security_level: u64,
) -> f64 {
    let v_keyswitch = variance_keyswitch(
        param.input_lwe_dimension.0,
        param.ks_decomposition_parameter.log2_base,
        param.ks_decomposition_parameter.level,
        ciphertext_modulus_log,
        minimal_variance_lwe(
            param.internal_lwe_dimension.0,
            ciphertext_modulus_log,
            security_level,
        ),
    );
    let v_modulus_switch = estimate_modulus_switching_noise_with_binary_key(
        param.internal_lwe_dimension.0,
        param.output_glwe_params.log2_polynomial_size,
        ciphertext_modulus_log,
    );
    input_variance + v_keyswitch + v_modulus_switch
}
