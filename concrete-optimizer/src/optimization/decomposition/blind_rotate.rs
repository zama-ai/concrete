use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::parameters::{BrDecompositionParameters, GlweParameters, LweDimension, PbsParameters};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::common::{MacroParam, VERSION};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BrComplexityNoise {
    pub decomp: BrDecompositionParameters,
    pub complexity: f64,
    pub noise: f64,
}

/* This is stricly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    security_level: u64,
    internal_dim: u64,
    glwe_params: GlweParameters,
) -> Vec<BrComplexityNoise> {
    assert!(ciphertext_modulus_log == 64);

    let variance_bsk = glwe_params.minimal_variance(ciphertext_modulus_log, security_level);

    let mut quantities = Vec::with_capacity(ciphertext_modulus_log as usize);
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut counting_no_progress = 0;

    let mut prev_best_log2_base = ciphertext_modulus_log as u64;

    for level in 1..=ciphertext_modulus_log as u64 {
        // detect increasing noise
        let mut level_decreasing_base_noise = f64::INFINITY;
        let mut best_log2_base = 0_u64;
        // we know a max is between 1 and prev_best_log2_base
        // and the curve has only 1 maximum close to prev_best_log2_base
        // so we start on prev_best_log2_base
        let range = (1..=prev_best_log2_base).rev();

        for log2_base in range {
            let base_noise = variance_blind_rotate(
                internal_dim,
                glwe_params.glwe_dimension,
                glwe_params.polynomial_size(),
                log2_base,
                level,
                ciphertext_modulus_log,
                variance_bsk,
            );
            if base_noise > level_decreasing_base_noise {
                break;
            }
            level_decreasing_base_noise = base_noise;
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

        let br_decomposition_parameter = BrDecompositionParameters {
            level,
            log2_base: best_log2_base,
        };
        let params = PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        let complexity_pbs = complexity_model.pbs_complexity(params, ciphertext_modulus_log);

        quantities.push(BrComplexityNoise {
            decomp: params.br_decomposition_parameter,
            noise: level_decreasing_base_noise,
            complexity: complexity_pbs,
        });
        assert!(increasing_complexity < complexity_pbs);
        increasing_complexity = complexity_pbs;
        decreasing_variance = level_decreasing_base_noise;
    }
    quantities
}

pub type Cache = CacheHashMap<MacroParam, Vec<BrComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(
        &mut self,
        glwe_params: GlweParameters,
        internal_dim: u64,
    ) -> &[BrComplexityNoise] {
        self.get((glwe_params, internal_dim))
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<MacroParam, Vec<BrComplexityNoise>>;

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let ciphertext_modulus_log = 64;
    let hardware = processing_unit.br_to_string();
    let path = format!("{cache_dir}/br-decomp-{hardware}-64-{security_level}");

    let function = move |(glwe_params, internal_dim): MacroParam| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            internal_dim,
            glwe_params,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
