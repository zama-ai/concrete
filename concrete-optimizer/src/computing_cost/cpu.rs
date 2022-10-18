use super::complexity::Complexity;
use super::complexity_model::ComplexityModel;
use super::operators::keyswitch_lwe::KsComplexity;
use super::operators::{keyswitch_lwe, pbs};
use crate::parameters::{KeyswitchParameters, LweDimension, PbsParameters};

#[derive(Clone)]
pub struct CpuComplexity {
    pub ks_lwe: keyswitch_lwe::KsComplexity,
    pub pbs: pbs::PbsComplexity,
}

impl ComplexityModel for CpuComplexity {
    fn pbs_complexity(&self, params: PbsParameters, ciphertext_modulus_log: u32) -> Complexity {
        self.pbs.complexity(params, ciphertext_modulus_log)
    }

    fn ks_complexity(
        &self,
        params: KeyswitchParameters,
        ciphertext_modulus_log: u32,
    ) -> Complexity {
        self.ks_lwe.complexity(params, ciphertext_modulus_log)
    }

    fn levelled_complexity(
        &self,
        sum_size: u64,
        lwe_dimension: LweDimension,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        sum_size as f64 * lwe_dimension.0 as f64
    }
}

impl Default for CpuComplexity {
    fn default() -> Self {
        Self {
            ks_lwe: KsComplexity,
            pbs: pbs::PbsComplexity::default(),
        }
    }
}
