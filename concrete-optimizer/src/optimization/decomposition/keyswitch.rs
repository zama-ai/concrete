use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use concrete_commons::dispersion::DispersionParameter;

use crate::computing_cost::operators::keyswitch_lwe::KsComplexity;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::parameters::{
    GlweParameters, KeyswitchParameters, KsDecompositionParameters, LweDimension,
};
use crate::security::security_weights::SECURITY_WEIGHTS_TABLE;
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::PersistentCacheHashMap;

use super::common::MacroParam;
use super::cut::ComplexityNoise;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct KsComplexityNoise {
    pub decomp: KsDecompositionParameters,
    pub complexity: f64,
    pub noise: f64,
}

impl ComplexityNoise for KsComplexityNoise {
    fn noise(&self) -> f64 {
        self.noise
    }
    fn complexity(&self) -> f64 {
        self.complexity
    }
}

/* This is stricly variance decreasing and strictly complexity increasing */
pub fn pareto_quantities(
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
    let mut prev_best_log2_base = 0_u64;
    let max_level = ciphertext_modulus_log as u64;
    for level in 1..=max_level {
        // detect increasing noise
        let mut level_decreasing_base_noise = f64::INFINITY;
        let mut best_log2_base = 0_u64;
        let range: Vec<_> = if level == 1 {
            (1..=(max_level / level)).collect()
        } else {
            // we know a max is between 1 and prev_best_log2_base
            // and the curve has only 1 maximum close to prev_best_log2_base
            // so we start on prev_best_log2_base
            (1..=prev_best_log2_base).rev().collect()
        };
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
        let complexity_keyswitch = KsComplexity.complexity(ks_params, ciphertext_modulus_log);
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

type PersistDecompCache = PersistentCacheHashMap<MacroParam, Vec<KsComplexityNoise>>;
type MultiSecPersistDecompCache = HashMap<u64, PersistDecompCache>;

#[static_init::dynamic]
pub static SHARED_CACHE: MultiSecPersistDecompCache = SECURITY_WEIGHTS_TABLE
    .keys()
    .map(|&security_level| {
        let ciphertext_modulus_log = 64;
        let tmp: String = std::env::temp_dir()
            .to_str()
            .expect("Invalid tmp dir")
            .into();
        let path = format!("{tmp}/optimizer/cache/ks-decomp-cpu-64-{security_level}");
        let function = move |(glwe_params, internal_dim): MacroParam| {
            pareto_quantities(
                ciphertext_modulus_log,
                security_level,
                internal_dim,
                glwe_params,
            )
        };
        (
            security_level,
            PersistentCacheHashMap::new(&path, "v0", function),
        )
    })
    .collect::<MultiSecPersistDecompCache>();

#[cfg(not(target_os = "macos"))]
#[static_init::destructor(10)]
extern "C" fn finaly() {
    for v in SHARED_CACHE.values() {
        v.sync_to_disk();
    }
}

pub fn for_security(security_level: u64) -> &'static PersistDecompCache {
    SHARED_CACHE.get(&security_level).unwrap()
}
