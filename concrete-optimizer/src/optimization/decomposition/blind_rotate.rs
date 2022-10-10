use std::sync::Arc;

use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::parameters::{BrDecompositionParameters, GlweParameters, LweDimension, PbsParameters};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::PersistentCacheHashMap;
use crate::{config, security};

use super::common::MacroParam;

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
    max_log2_base: u64,
) -> Vec<BrComplexityNoise> {
    assert!(ciphertext_modulus_log == 64);
    let pbs_param = |level, log2_base| {
        let br_decomposition_parameter = BrDecompositionParameters { level, log2_base };
        PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        }
    };
    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);

    let mut quantities = Vec::with_capacity(max_log2_base as usize);
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut counting_no_progress = 0;

    let mut prev_best_log2_base = max_log2_base;

    for level in 1..=ciphertext_modulus_log as u64 {
        // detect increasing noise
        let mut level_decreasing_base_noise = f64::INFINITY;
        let mut best_log2_base = 0_u64;
        // we know a max is between 1 and prev_best_log2_base
        // and the curve has only 1 maximum close to prev_best_log2_base
        // so we start on prev_best_log2_base
        let range = (1..=prev_best_log2_base).rev();

        for log2_base in range {
            let base_noise = noise_atomic_pattern::variance_bootstrap(
                pbs_param(level, log2_base),
                ciphertext_modulus_log,
                variance_bsk,
            )
            .get_variance();
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
        let params = pbs_param(level, best_log2_base);
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
    let max_log2_base = processing_unit.max_br_base_log();

    let ciphertext_modulus_log = 64;
    let tmp: String = std::env::temp_dir()
        .to_str()
        .expect("Invalid tmp dir")
        .into();

    let hardware = processing_unit.br_to_string();

    let path = format!("{tmp}/optimizer/cache/br-decomp-{hardware}-64-{security_level}");

    let function = move |(glwe_params, internal_dim): MacroParam| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            internal_dim,
            glwe_params,
            max_log2_base,
        )
    };
    PersistentCacheHashMap::new(&path, "v0", function)
}
