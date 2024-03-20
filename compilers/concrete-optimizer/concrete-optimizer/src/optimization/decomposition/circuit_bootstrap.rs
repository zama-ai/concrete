use std::sync::Arc;

use concrete_cpu_noise_model::gaussian_noise::noise::cmux::variance_cmux;
use serde::{Deserialize, Serialize};

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::parameters::{BrDecompositionParameters, CmuxParameters, GlweParameters};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};
use crate::utils::square;

use super::common::VERSION;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CbComplexityNoise {
    pub decomp: BrDecompositionParameters,
    pub complexity_one_cmux_hp: f64,
    pub complexity_one_ggsw_to_fft: f64,
    pub variance_bias: f64,
    pub variance_ggsw_factor: f64,
}

impl CbComplexityNoise {
    pub fn variance_from_ggsw(&self, variance_ggsw: f64) -> f64 {
        self.variance_bias + self.variance_ggsw_factor * variance_ggsw
    }
}

/* This is strictly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
    glwe_params: GlweParameters,
) -> Vec<CbComplexityNoise> {
    let cmux_param = |level, log2_base| {
        let br_decomposition_parameter = BrDecompositionParameters { level, log2_base };
        CmuxParameters {
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        }
    };
    let mut quantities = Vec::with_capacity(ciphertext_modulus_log as usize);
    let max_level = ciphertext_modulus_log as u64;
    for level in 1..=max_level {
        // detect increasing noise
        for log2_base in 1..=(max_level / level) {
            let params = cmux_param(level, log2_base);
            // Hybrid packing
            let complexity_one_cmux_hp =
                complexity_model.cmux_complexity(params, ciphertext_modulus_log);

            let f_glwe_poly_size = glwe_params.polynomial_size() as f64;

            let f_glwe_size = (glwe_params.glwe_dimension + 1) as f64;

            let complexity_one_ggsw_to_fft = square(f_glwe_size)
                * level as f64
                * complexity_model.fft_complexity(f_glwe_poly_size, ciphertext_modulus_log);

            // Compute bias and slove for variance_one_external_product_for_cmux_tree_bias
            let variance = |variance_bsk| {
                variance_cmux(
                    glwe_params.glwe_dimension,
                    glwe_params.polynomial_size(),
                    log2_base,
                    level,
                    ciphertext_modulus_log,
                    fft_precision,
                    variance_bsk,
                )
            };
            let variance_at_0 = variance(0.0);
            let variance_at_1 = variance(1.0);
            let variance_one_external_product_for_cmux_tree_bias = variance_at_0;
            let variance_one_external_product_for_cmux_tree_slope = variance_at_1 - variance_at_0;

            quantities.push(CbComplexityNoise {
                decomp: params.br_decomposition_parameter,
                complexity_one_cmux_hp,
                complexity_one_ggsw_to_fft,
                variance_bias: variance_one_external_product_for_cmux_tree_bias,
                variance_ggsw_factor: variance_one_external_product_for_cmux_tree_slope,
            });
        }
    }
    quantities
}

pub type Cache = CacheHashMap<GlweParameters, CbPareto>;

impl Cache {
    pub fn pareto_quantities(&mut self, glwe_params: GlweParameters) -> &CbPareto {
        self.get(glwe_params)
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<GlweParameters, CbPareto>;

#[derive(Clone, Serialize, Deserialize)]
pub struct CbPareto {
    pub pareto: Vec<CbComplexityNoise>,
    pub lower_pareto_cb_bias: f64,
    pub lower_pareto_cb_slope: f64,
    pub lower_bound_cost_cb_complexity_1_cmux_hp: f64,
    pub lower_bound_cost_cb_complexity_1_ggsw_to_fft: f64,
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let hardware = processing_unit.br_to_string();
    let path =
        format!("{cache_dir}/cb-decomp-{hardware}-{ciphertext_modulus_log}-{fft_precision}-{security_level}");
    let function = move |glwe_params| {
        let pareto = pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            fft_precision,
            glwe_params,
        );

        let lower_pareto_cb_bias = pareto
            .iter()
            .map(|cb| cb.variance_bias)
            .reduce(f64::min)
            .unwrap();
        let lower_pareto_cb_slope = pareto
            .iter()
            .map(|cb| cb.variance_ggsw_factor)
            .reduce(f64::min)
            .unwrap();
        let lower_bound_cost_cb_complexity_1_cmux_hp = pareto
            .iter()
            .map(|cb| cb.complexity_one_cmux_hp)
            .reduce(f64::min)
            .unwrap();
        let lower_bound_cost_cb_complexity_1_ggsw_to_fft = pareto
            .iter()
            .map(|cb| cb.complexity_one_ggsw_to_fft)
            .reduce(f64::min)
            .unwrap();

        CbPareto {
            pareto,
            lower_pareto_cb_bias,
            lower_pareto_cb_slope,
            lower_bound_cost_cb_complexity_1_cmux_hp,
            lower_bound_cost_cb_complexity_1_ggsw_to_fft,
        }
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
