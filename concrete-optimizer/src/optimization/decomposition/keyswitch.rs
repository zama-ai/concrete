use std::sync::Arc;

use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::parameters::{
    GlweParameters, KeyswitchParameters, KsDecompositionParameters, LweDimension,
};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};

use super::common::{MacroParam, VERSION};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KsComplexityNoise {
    pub decomp: KsDecompositionParameters,
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
) -> Vec<KsComplexityNoise> {
    assert!(ciphertext_modulus_log == 64);
    let glwe_poly_size = glwe_params.polynomial_size();
    let input_lwe_dimension = glwe_params.glwe_dimension * glwe_poly_size;
    let ks_param = |level, log2_base| {
        let ks_decomposition_parameter = KsDecompositionParameters { level, log2_base };
        KeyswitchParameters {
            input_lwe_dimension: LweDimension(input_lwe_dimension),
            output_lwe_dimension: LweDimension(internal_dim),
            ks_decomposition_parameter,
        }
    };
    let variance_ksk =
        noise_atomic_pattern::variance_ksk(internal_dim, ciphertext_modulus_log, security_level);

    let mut quantities = Vec::with_capacity(64);
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
            let noise_keyswitch = noise_atomic_pattern::variance_keyswitch(
                ks_param(level, log2_base),
                ciphertext_modulus_log,
                variance_ksk,
            )
            .get_variance();
            if noise_keyswitch > level_decreasing_base_noise {
                break;
            }
            level_decreasing_base_noise = noise_keyswitch;
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
        let ks_params = ks_param(level, best_log2_base);
        let complexity_keyswitch =
            complexity_model.ks_complexity(ks_params, ciphertext_modulus_log);
        quantities.push(KsComplexityNoise {
            decomp: ks_params.ks_decomposition_parameter,
            noise: level_decreasing_base_noise,
            complexity: complexity_keyswitch,
        });
        assert!(increasing_complexity < complexity_keyswitch);
        increasing_complexity = complexity_keyswitch;
        decreasing_variance = level_decreasing_base_noise;
    }
    quantities
}

pub type Cache = CacheHashMap<MacroParam, Vec<KsComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(
        &mut self,
        glwe_params: GlweParameters,
        internal_dim: u64,
    ) -> &[KsComplexityNoise] {
        self.get((glwe_params, internal_dim))
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<MacroParam, Vec<KsComplexityNoise>>;

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let ciphertext_modulus_log = 64;
    let hardware = processing_unit.ks_to_string();
    let path = format!("{cache_dir}/ks-decomp-{hardware}-64-{security_level}");

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
