use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::parameters::{BrDecompositionParameters, CmuxParameters, GlweParameters};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};
use concrete_cpu_noise_model::gaussian_noise::noise::cmux::variance_cmux;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::common::VERSION;
use super::DecompCaches;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CmuxComplexityNoise {
    pub decomp: BrDecompositionParameters,
    pub complexity: f64,
    pub noise: f64,
}

impl CmuxComplexityNoise {
    pub fn complexity_br(&self, in_lwe_dim: u64) -> f64 {
        in_lwe_dim as f64 * self.complexity
    }
    pub fn noise_br(&self, in_lwe_dim: u64) -> f64 {
        in_lwe_dim as f64 * self.noise
    }
}

/* This is strictly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
    security_level: u64,
    glwe_params: GlweParameters,
) -> Vec<CmuxComplexityNoise> {
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
            let base_noise = variance_cmux(
                glwe_params.glwe_dimension,
                glwe_params.polynomial_size(),
                log2_base,
                level,
                ciphertext_modulus_log,
                fft_precision,
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
        let params = CmuxParameters {
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        let complexity = complexity_model.cmux_complexity(params, ciphertext_modulus_log);

        quantities.push(CmuxComplexityNoise {
            decomp: params.br_decomposition_parameter,
            noise: level_decreasing_base_noise,
            complexity,
        });
        assert!(increasing_complexity < complexity);
        increasing_complexity = complexity;
        decreasing_variance = level_decreasing_base_noise;
    }
    quantities
}

pub fn lowest_noise(quantities: &[CmuxComplexityNoise]) -> CmuxComplexityNoise {
    quantities[quantities.len() - 1]
}

pub fn lowest_noise_br(quantities: &[CmuxComplexityNoise], in_lwe_dim: u64) -> f64 {
    lowest_noise(quantities).noise_br(in_lwe_dim)
}

pub fn lowest_complexity_br(quantities: &[CmuxComplexityNoise], in_lwe_dim: u64) -> f64 {
    quantities[0].complexity_br(in_lwe_dim)
}

pub type Cache = CacheHashMap<GlweParameters, Vec<CmuxComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(&mut self, glwe_params: GlweParameters) -> &[CmuxComplexityNoise] {
        self.get(glwe_params)
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<GlweParameters, Vec<CmuxComplexityNoise>>;

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
        format!("{cache_dir}/cmux-decomp-{hardware}-{ciphertext_modulus_log}-{fft_precision}-{security_level}");

    let function = move |glwe_params: GlweParameters| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            fft_precision,
            security_level,
            glwe_params,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}

#[derive(Debug)]
pub enum MaxVarianceError {
    PbsBaseLogNotFound,
    PbsLevelNotFound,
}

pub fn get_noise_br(
    mut cache: DecompCaches,
    log2_polynomial_size: u64,
    glwe_dimension: u64,
    lwe_dim: u64,
    pbs_level: u64,
    pbs_log2_base: Option<u64>,
) -> Result<f64, MaxVarianceError> {
    let cmux_quantities = cache.cmux.pareto_quantities(GlweParameters {
        log2_polynomial_size,
        glwe_dimension,
    });
    for cmux_quantity in cmux_quantities {
        if cmux_quantity.decomp.level == pbs_level {
            if pbs_log2_base.is_some() && cmux_quantity.decomp.log2_base != pbs_log2_base.unwrap() {
                return Err(MaxVarianceError::PbsBaseLogNotFound);
            }
            return Ok(cmux_quantity.noise_br(lwe_dim));
        }
    }
    Err(MaxVarianceError::PbsLevelNotFound)
}
