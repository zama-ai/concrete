use super::common::VERSION;
use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::parameters::{KeyswitchParameters, KsDecompositionParameters, LweDimension};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::{default_cache_dir, PersistentCacheHashMap};
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch_one_bit::variance_keyswitch_one_bit;
use concrete_security_curves::gaussian::security::minimal_variance_lwe;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KsComplexityNoise {
    pub decomp: KsDecompositionParameters,
    pub complexity_bias: f64,
    pub complexity_slope: f64,
    pub noise: f64,
}

impl KsComplexityNoise {
    pub fn complexity(&self, in_lwe_dim: u64) -> f64 {
        self.complexity_bias + in_lwe_dim as f64 * self.complexity_slope
    }
    pub fn noise(&self, in_lwe_dim: u64) -> f64 {
        in_lwe_dim as f64 * self.noise
    }
}

/* This is strictly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    security_level: u64,
    internal_dim: u64,
) -> Vec<KsComplexityNoise> {
    let variance_ksk = minimal_variance_lwe(internal_dim, ciphertext_modulus_log, security_level);

    let mut quantities = Vec::with_capacity(ciphertext_modulus_log as usize);
    let mut increasing_complexity_slope = 0.0;
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
            let noise_keyswitch =
                variance_keyswitch_one_bit(log2_base, level, ciphertext_modulus_log, variance_ksk);
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

        let decomp = KsDecompositionParameters {
            level,
            log2_base: best_log2_base,
        };

        let complexity_at = |in_lwe_dim| {
            complexity_model.ks_complexity(
                KeyswitchParameters {
                    input_lwe_dimension: LweDimension(in_lwe_dim),
                    output_lwe_dimension: LweDimension(internal_dim),
                    ks_decomposition_parameter: decomp,
                },
                ciphertext_modulus_log,
            )
        };

        let complexity_at_0 = complexity_at(0);
        let complexity_at_1 = complexity_at(1);

        let complexity_bias = complexity_at_0;

        let complexity_slope = complexity_at_1 - complexity_at_0;

        quantities.push(KsComplexityNoise {
            decomp,
            noise: level_decreasing_base_noise,
            complexity_slope,
            complexity_bias,
        });
        assert!(increasing_complexity_slope < complexity_slope);
        increasing_complexity_slope = complexity_slope;
        decreasing_variance = level_decreasing_base_noise;
    }
    quantities
}

pub fn lowest_noise_ks(quantities: &[KsComplexityNoise], in_lwe_dim: u64) -> f64 {
    quantities[quantities.len() - 1].noise(in_lwe_dim)
}

pub fn lowest_complexity_ks(quantities: &[KsComplexityNoise], in_lwe_dim: u64) -> f64 {
    quantities[0].complexity(in_lwe_dim)
}

pub type Cache = CacheHashMap<u64, Vec<KsComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(&mut self, internal_dim: u64) -> &[KsComplexityNoise] {
        self.get(internal_dim)
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<u64, Vec<KsComplexityNoise>>;

pub fn cache(
    security_level: u64,
    processing_unit: config::ProcessingUnit,
    complexity_model: Arc<dyn ComplexityModel>,
    ciphertext_modulus_log: u32,
) -> PersistDecompCache {
    let cache_dir: String = default_cache_dir();
    let hardware = processing_unit.ks_to_string();
    let path =
        format!("{cache_dir}/ks-decomp-{hardware}-{ciphertext_modulus_log}-{security_level}");

    let function = move |internal_dim: u64| {
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            internal_dim,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
