use super::common::VERSION;
use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::parameters::{
    BrDecompositionParameters, GlweParameters, KeyswitchParameters, KsDecompositionParameters,
    LweDimension,
};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};
use concrete_cpu_noise_model::gaussian_noise::noise::private_packing_keyswitch::estimate_packing_private_keyswitch;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpSwitchComplexityNoise {
    pub decomp: BrDecompositionParameters,
    pub complexity: f64,
    pub noise: f64,
}

pub type Cache = CacheHashMap<GlweParameters, Vec<PpSwitchComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(&mut self, glwe_params: GlweParameters) -> &[PpSwitchComplexityNoise] {
        self.get(glwe_params)
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<GlweParameters, Vec<PpSwitchComplexityNoise>>;

pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    security_level: u64,
    glwe_params: GlweParameters,
) -> Vec<PpSwitchComplexityNoise> {
    let variance_bsk = glwe_params.minimal_variance(ciphertext_modulus_log, security_level);
    let mut quantities = Vec::with_capacity(ciphertext_modulus_log as usize);
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut counting_no_progress = 0;

    let max_level = ciphertext_modulus_log as u64;

    let mut prev_best_log2_base = ciphertext_modulus_log as u64;
    for level in 1..=max_level {
        // detect increasing noise
        let mut level_decreasing_base_noise = f64::INFINITY;
        let mut best_log2_base = 0_u64;

        // we know a max is between 1 and prev_best_log2_base
        // and the curve has only 1 maximum close to prev_best_log2_base
        // so we start on prev_best_log2_base
        let max_log2_base = prev_best_log2_base.min(max_level / level);
        for log2_base in (1..=max_log2_base).rev() {
            let variance_private_packing_ks = estimate_packing_private_keyswitch(
                0.,
                variance_bsk,
                log2_base,
                level,
                glwe_params.glwe_dimension,
                glwe_params.polynomial_size(),
                ciphertext_modulus_log,
            );
            if variance_private_packing_ks > level_decreasing_base_noise {
                break;
            }
            level_decreasing_base_noise = variance_private_packing_ks;
            best_log2_base = log2_base;
        }
        prev_best_log2_base = best_log2_base;
        if decreasing_variance < level_decreasing_base_noise {
            // the current case is dominated
            if best_log2_base == 1 {
                counting_no_progress += 1;
                if counting_no_progress > 16 {
                    break;
                }
            }
            continue;
        }
        let sample_extract_lwe_dimension = LweDimension(glwe_params.sample_extract_lwe_dimension());
        let ppks_parameter_complexity = KeyswitchParameters {
            input_lwe_dimension: sample_extract_lwe_dimension,
            output_lwe_dimension: sample_extract_lwe_dimension,
            ks_decomposition_parameter: KsDecompositionParameters {
                level,
                log2_base: best_log2_base,
            },
        };
        let complexity_ppks =
            complexity_model.ks_complexity(ppks_parameter_complexity, ciphertext_modulus_log);
        let pp_ks_decomposition_parameter = BrDecompositionParameters {
            level,
            log2_base: best_log2_base,
        };
        quantities.push(PpSwitchComplexityNoise {
            decomp: pp_ks_decomposition_parameter,
            complexity: complexity_ppks,
            noise: level_decreasing_base_noise,
        });
        assert!(increasing_complexity < complexity_ppks);
        increasing_complexity = complexity_ppks;
        decreasing_variance = level_decreasing_base_noise;
    }
    quantities
}

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
    ciphertext_modulus_log: u32,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let hardware = processing_unit.br_to_string();
    let path =
        format!("{cache_dir}/pp-decomp-{hardware}-{ciphertext_modulus_log}-{security_level}");

    let function = move |glwe_params: GlweParameters| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            glwe_params,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
