use super::complexity::Complexity;
use super::complexity_model::ComplexityModel;
use super::operators::keyswitch_lwe::KsComplexity;
use super::operators::{keyswitch_lwe, multi_bit_pbs, pbs};
use crate::computing_cost::operators::multi_bit_pbs::MultiBitPbsComplexity;
use crate::parameters::{CmuxParameters, KeyswitchParameters, LweDimension, PbsParameters};

#[derive(Clone)]
pub struct CpuComplexity {
    pub ks_lwe: keyswitch_lwe::KsComplexity,
    pub pbs: pbs::PbsComplexity,
    pub multi_bit_pbs: MultiBitPbsComplexity,
}

impl ComplexityModel for CpuComplexity {
    fn pbs_complexity(&self, params: PbsParameters, ciphertext_modulus_log: u32) -> Complexity {
        self.pbs.complexity(params, ciphertext_modulus_log)
    }
    fn multi_bit_pbs_complexity(
        &self,
        params: PbsParameters,
        ciphertext_modulus_log: u32,
        grouping_factor: u32,
        jit_fft: bool,
    ) -> Complexity {
        self.multi_bit_pbs
            .complexity(params, ciphertext_modulus_log, grouping_factor, jit_fft)
    }

    fn cmux_complexity(&self, params: CmuxParameters, ciphertext_modulus_log: u32) -> Complexity {
        self.pbs.cmux.complexity(params, ciphertext_modulus_log)
    }

    fn ks_complexity(
        &self,
        params: KeyswitchParameters,
        ciphertext_modulus_log: u32,
    ) -> Complexity {
        self.ks_lwe.complexity(params, ciphertext_modulus_log)
    }

    fn fft_complexity(&self, glwe_polynomial_size: f64, ciphertext_modulus_log: u32) -> Complexity {
        self.pbs
            .cmux
            .fft_complexity(glwe_polynomial_size, ciphertext_modulus_log)
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
            multi_bit_pbs: multi_bit_pbs::MultiBitPbsComplexity::default(),
        }
    }
}
