use std::sync::Arc;

use concrete_commons::dispersion::{DispersionParameter, Variance};
use serde::{Deserialize, Serialize};

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::noise_estimator::operators::atomic_pattern::variance_cmux;
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

/* This is stricly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    glwe_params: GlweParameters,
) -> Vec<CbComplexityNoise> {
    assert!(ciphertext_modulus_log == 64);
    let cmux_param = |level, log2_base| {
        let br_decomposition_parameter = BrDecompositionParameters { level, log2_base };
        CmuxParameters {
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        }
    };
    let mut quantities = Vec::with_capacity(64);
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
                let variance_bsk = Variance::from_variance(variance_bsk);
                variance_cmux(params, ciphertext_modulus_log, variance_bsk).get_variance()
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

pub type Cache = CacheHashMap<GlweParameters, Vec<CbComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(&mut self, glwe_params: GlweParameters) -> &[CbComplexityNoise] {
        self.get(glwe_params)
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<GlweParameters, Vec<CbComplexityNoise>>;

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let ciphertext_modulus_log = 64;
    let hardware = processing_unit.br_to_string();
    let path =
        format!("{cache_dir}/cb-decomp-{hardware}-{ciphertext_modulus_log}-{security_level}");
    let function = move |glwe_params| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            glwe_params,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
