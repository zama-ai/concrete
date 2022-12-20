use super::cmux;
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
    cmux_quantities: &[cmux::CmuxComplexityNoise],
) -> Vec<PpSwitchComplexityNoise> {
    let variance_bsk = glwe_params.minimal_variance(ciphertext_modulus_log, security_level);
    let mut result = Vec::with_capacity(cmux_quantities.len());
    for cmux in cmux_quantities {
        let pp_ks_decomposition_parameter = cmux.decomp;

        // We assume the packing KS and the external product in a PBSto have
        // the same parameters (base, level)
        let variance_private_packing_ks = estimate_packing_private_keyswitch(
            0.,
            variance_bsk,
            pp_ks_decomposition_parameter.log2_base,
            pp_ks_decomposition_parameter.level,
            glwe_params.glwe_dimension,
            glwe_params.polynomial_size(),
            ciphertext_modulus_log,
        );

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
            decomp: cmux.decomp,
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
    let cache_dir: String = default_cache_dir();
    let ciphertext_modulus_log = 64;
    let hardware = processing_unit.br_to_string();
    let path = format!("{cache_dir}/bc-decomp-{hardware}-64-{security_level}");

    let function = move |glwe_params: GlweParameters| {
        let br = cmux::pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            glwe_params,
        );
        pareto_quantities(
            complexity_model.as_ref(),
            ciphertext_modulus_log,
            security_level,
            glwe_params,
            &br,
        )
    };
    PersistentCacheHashMap::new_no_read(&path, VERSION, function)
}
