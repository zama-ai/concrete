use std::sync::Arc;

use concrete_commons::dispersion::{DispersionParameter, Variance};
use serde::{Deserialize, Serialize};

use crate::computing_cost::complexity_model::ComplexityModel;

use crate::noise_estimator::operators::wop_atomic_pattern::estimate_packing_private_keyswitch;
use crate::parameters::{
    BrDecompositionParameters, CmuxParameters, GlweParameters, KeyswitchParameters,
    KsDecompositionParameters, LweDimension,
};
use crate::utils::cache::ephemeral::{CacheHashMap, EphemeralCache};
use crate::utils::cache::persistent::PersistentCacheHashMap;
use crate::{config, security};

use super::blind_rotate;
use super::common::MacroParam;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PpSwitchComplexityNoise {
    pub decomp: BrDecompositionParameters,
    pub complexity: f64,
    pub noise: f64,
}

pub type Cache = CacheHashMap<MacroParam, Vec<PpSwitchComplexityNoise>>;

impl Cache {
    pub fn pareto_quantities(
        &mut self,
        glwe_params: GlweParameters,
        internal_dim: u64,
    ) -> &[PpSwitchComplexityNoise] {
        self.get((glwe_params, internal_dim))
    }
}

pub type PersistDecompCache = PersistentCacheHashMap<MacroParam, Vec<PpSwitchComplexityNoise>>;

pub fn pareto_quantities(
    complexity_model: &dyn ComplexityModel,
    ciphertext_modulus_log: u32,
    security_level: u64,
    _internal_dim: u64,
    glwe_params: GlweParameters,
    br_quantities: &[blind_rotate::BrComplexityNoise],
) -> Vec<PpSwitchComplexityNoise> {
    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);
    let mut result = Vec::with_capacity(br_quantities.len());
    for br in br_quantities {
        let pp_ks_decomposition_parameter = br.decomp;
        let ppks_parameter = CmuxParameters {
            br_decomposition_parameter: pp_ks_decomposition_parameter,
            output_glwe_params: glwe_params,
        };
        // We assume the packing KS and the external product in a PBSto have
        // the same parameters (base, level)
        let variance_private_packing_ks = estimate_packing_private_keyswitch::<u64>(
            Variance(0.),
            variance_bsk,
            ppks_parameter,
            ciphertext_modulus_log,
        )
        .get_variance();

        let sample_extract_lwe_dimension = LweDimension(glwe_params.sample_extract_lwe_dimension());
        let ppks_parameter_complexity = KeyswitchParameters {
            input_lwe_dimension: sample_extract_lwe_dimension,
            output_lwe_dimension: sample_extract_lwe_dimension,
            ks_decomposition_parameter: KsDecompositionParameters {
                level: pp_ks_decomposition_parameter.level,
                log2_base: pp_ks_decomposition_parameter.log2_base,
            },
        };
        let complexity_ppks =
            complexity_model.ks_complexity(ppks_parameter_complexity, ciphertext_modulus_log);
        result.push(PpSwitchComplexityNoise {
            decomp: br.decomp,
            complexity: complexity_ppks,
            noise: variance_private_packing_ks,
        });
    }
    result
}

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
    let path = format!("{tmp}/optimizer/cache/bc-decomp-{hardware}-64-{security_level}");

    let function = move |(glwe_params, internal_dim): MacroParam| {
        let br = blind_rotate::pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            internal_dim,
            glwe_params,
            max_log2_base,
        );
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            internal_dim,
            glwe_params,
            &br,
        )
    };
    PersistentCacheHashMap::new(&path, "v0", function)
}
