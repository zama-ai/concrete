use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_npe::KeyDispersion;

use crate::parameters::CmuxParameters;
use crate::utils::square;

pub fn estimate_packing_private_keyswitch<T>(
    var_glwe: Variance,
    var_ggsw: Variance,
    param: CmuxParameters,
    ciphertext_modulus_log: u32,
) -> Variance {
    type K = BinaryKeyKind;
    let l = param.br_decomposition_parameter.level as f64;
    let b = (1 << param.br_decomposition_parameter.log2_base) as f64;
    let n = (param.output_glwe_params.glwe_dimension * param.output_glwe_params.polynomial_size())
        as f64; // param.internal_lwe_dimension.0 as f64;
    let b2l = f64::powf(b, 2. * l);
    let var_s_w = 1. / 4.;
    let mean_s_w = 1. / 2.;
    // println!("n = {}", n);
    let res_1 = (l * (n + 1.) * var_ggsw.get_modular_variance(ciphertext_modulus_log)) as f64
        * (square(b) + 2.)
        / 12.;

    #[allow(clippy::cast_possible_wrap)]
    let res_3 = ((square(f64::powi(2., ciphertext_modulus_log as i32)) as f64 - b2l) / (12. * b2l)
        * (1.
            + n * (K::variance_key_coefficient(ciphertext_modulus_log)
                .get_modular_variance(ciphertext_modulus_log)
                + square(K::expectation_key_coefficient())))
        + n / 4.
            * K::variance_key_coefficient(ciphertext_modulus_log)
                .get_modular_variance(ciphertext_modulus_log)
        + var_glwe.get_modular_variance(ciphertext_modulus_log))
        * (var_s_w + square(mean_s_w));

    let res_5 = var_s_w * (1. / 4. * square(1. - n * K::expectation_key_coefficient()));

    Variance::from_modular_variance(res_1 + res_3 + res_5, ciphertext_modulus_log)
}
