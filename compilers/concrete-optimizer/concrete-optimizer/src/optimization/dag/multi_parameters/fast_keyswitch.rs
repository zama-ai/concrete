// TODO: move to cache with pareto check

use concrete_cpu_noise_model::gaussian_noise::conversion::modular_variance_to_variance;

// TODO: move to concrete-cpu
use crate::optimization::decomposition::keyswitch::KsComplexityNoise;
use crate::parameters::{GlweParameters, KsDecompositionParameters};

use serde::{Deserialize, Serialize};

// output glwe is incorporated in KsComplexityNoise
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FksComplexityNoise {
    // 1 -> 0
    // k1⋅N1>k0⋅N0, k0⋅N0≥k1⋅N1
    pub decomp: KsDecompositionParameters,
    pub noise: f64,
    pub complexity: f64,
    pub src_glwe_param: GlweParameters,
    pub dst_glwe_param: GlweParameters,
}

// Copy & paste from concrete-cpu
const FFT_SCALING_WEIGHT: f64 = -2.577_224_94;
fn fft_noise_variance_external_product_glwe(
    glwe_dimension: u64,
    polynomial_size: u64,
    log2_base: u64,
    level: u64,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> f64 {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L25
    let b = 2_f64.powi(log2_base as i32);
    let l = level as f64;
    let big_n = polynomial_size as f64;
    let k = glwe_dimension;
    assert!(k > 0, "k = {k}");

    #[allow(clippy::cast_possible_wrap)]
    let lost_bits = ciphertext_modulus_log as i32 - fft_precision as i32;

    let scale_margin = 2_f64.powi(2 * lost_bits);

    let res =
        f64::exp2(FFT_SCALING_WEIGHT) * scale_margin * l * b * b * big_n.powi(2) * (k as f64 + 1.);
    modular_variance_to_variance(res, ciphertext_modulus_log)
}

#[allow(non_snake_case)]
fn upper_k0(input_glwe: &GlweParameters, output_glwe: &GlweParameters) -> u64 {
    let k1 = input_glwe.glwe_dimension;
    let N1 = input_glwe.polynomial_size();
    let k0 = output_glwe.glwe_dimension;
    let N0 = output_glwe.polynomial_size();
    assert!(k1 * N1 >= k0 * N0);
    // candidate * N0 >= k1 * N1
    let f_upper_k0 = (k1 * N1) as f64 / N0 as f64;
    #[allow(clippy::cast_sign_loss)]
    let upper_k0 = f_upper_k0.ceil() as u64;
    upper_k0
}

#[allow(non_snake_case)]
pub fn complexity(input_glwe: &GlweParameters, output_glwe: &GlweParameters, level: u64) -> f64 {
    let k0 = output_glwe.glwe_dimension;
    let N0 = output_glwe.polynomial_size();
    let upper_k0 = upper_k0(input_glwe, output_glwe);
    #[allow(clippy::cast_sign_loss)]
    let log2_N0 = (N0 as f64).log2().ceil() as u64;
    let size0 = (k0 + 1) * N0 * log2_N0;
    let mul_count = size0 * upper_k0 * level;
    let add_count = size0 * (upper_k0 * level - 1);
    (add_count + mul_count) as f64
}

#[allow(non_snake_case)]
pub fn noise(
    ks: &KsComplexityNoise,
    input_glwe: &GlweParameters,
    output_glwe: &GlweParameters,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> f64 {
    let N0 = output_glwe.polynomial_size();
    let upper_k0 = upper_k0(input_glwe, output_glwe);
    ks.noise(input_glwe.sample_extract_lwe_dimension())
        + fft_noise_variance_external_product_glwe(
            upper_k0,
            N0,
            ks.decomp.log2_base,
            ks.decomp.level,
            ciphertext_modulus_log,
            fft_precision,
        )
}
