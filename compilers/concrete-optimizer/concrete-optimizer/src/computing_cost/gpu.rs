use super::complexity::Complexity;
use super::complexity_model::ComplexityModel;
use crate::parameters::{CmuxParameters, KeyswitchParameters, LweDimension, PbsParameters};
use crate::utils::square;

#[derive(Clone, Copy)]
pub enum GpuPbsComplexity {
    Lowlat,
    Amortized,
}

#[derive(Clone, Copy)]
pub struct GpuKsComplexity;

#[derive(Clone, Copy)]
pub struct GpuComplexity {
    pub ks: GpuKsComplexity,
    pub pbs: GpuPbsComplexity,
    pub number_of_sm: u64,
}

impl GpuComplexity {
    pub fn default_lowlat_u64(number_of_sm: u64) -> Self {
        Self {
            ks: GpuKsComplexity,
            pbs: GpuPbsComplexity::Lowlat,
            number_of_sm,
        }
    }

    pub fn default_amortized_u64(number_of_sm: u64) -> Self {
        Self {
            ks: GpuKsComplexity,
            pbs: GpuPbsComplexity::Amortized,
            number_of_sm,
        }
    }
}

impl ComplexityModel for GpuComplexity {
    #[allow(clippy::let_and_return, non_snake_case)]
    fn pbs_complexity(&self, _params: PbsParameters, _ciphertext_modulus_log: u32) -> Complexity {
        todo!()
    }

    fn cmux_complexity(&self, _params: CmuxParameters, _ciphertext_modulus_log: u32) -> Complexity {
        todo!()
    }

    #[allow(clippy::let_and_return)]
    fn ks_complexity(
        &self,
        _params: KeyswitchParameters,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        todo!()
    }

    fn fft_complexity(
        &self,
        _glwe_polynomial_size: f64,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        todo!()
    }

    fn levelled_complexity(
        &self,
        _sum_size: u64,
        _lwe_dimension: LweDimension,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        0.
    }

    fn multi_bit_pbs_complexity(
        &self,
        _params: PbsParameters,
        _ciphertext_modulus_log: u32,
        _grouping_factor: u32,
        _jit_fft: bool,
    ) -> Complexity {
        todo!()
    }
}

#[allow(non_snake_case)]
#[allow(dead_code)]
fn algorithmic_complexity_pbs(n: f64, k: f64, N: f64, ell: f64) -> f64 {
    n * (ell * (k + 1.) * N * (N.log2() + 1.)
        + (k + 1.) * N * (N.log2() + 1.)
        + N * ell * square(k + 1.))
}

#[allow(non_snake_case)]
#[allow(dead_code)]
fn algorithmic_complexity_ks(na: f64, nb: f64, ell: f64, log2_q: f64) -> f64 {
    na * nb * ell * log2_q
}
