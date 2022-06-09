use super::complexity::Complexity;
use super::complexity_model::ComplexityModel;
use crate::parameters::{KeyswitchParameters, LweDimension, PbsParameters};
use crate::utils::square;

#[derive(Clone, Copy)]
pub struct GpuPbsComplexity {
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub occupancy: f64,
}

//https://github.com/zama-ai/concrete-core-internal/issues/91
impl GpuPbsComplexity {
    pub fn default_lowlat_u64(occupancy: f64) -> Self {
        Self {
            w1: 2_576.105_013_4,
            w2: -21_631.382_229_52,
            w3: -86_525.527_535_17,
            w4: 0.125_472_398_538_904_43,
            occupancy,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GpuKsComplexity {
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub occupancy: f64,
    pub number_of_sm: u64,
}

// https://github.com/zama-ai/concrete-core-internal/issues/90
impl GpuKsComplexity {
    pub fn default_u64(occupancy: f64, number_of_sm: u64) -> Self {
        Self {
            w1: 7_959.869_676_54,
            w2: 3_866.817_732_87,
            w3: 8_353.484_127_44,
            w4: 0.125_472_398_538_904_43,
            occupancy,
            number_of_sm,
        }
    }
}

#[derive(Clone, Copy)]
pub struct GpuComplexity {
    pub ks: GpuKsComplexity,
    pub pbs: GpuPbsComplexity,
    pub ncores: u64,
}

impl ComplexityModel for GpuComplexity {
    #[allow(clippy::let_and_return, non_snake_case)]
    fn pbs_complexity(&self, params: PbsParameters, _ciphertext_modulus_log: u32) -> Complexity {
        let GpuPbsComplexity {
            w1,
            w2,
            w3,
            w4,
            occupancy,
        } = self.pbs;

        let n = params.internal_lwe_dimension.0 as f64;
        let k = params.output_glwe_params.glwe_dimension as f64;
        let N = (1 << params.output_glwe_params.log2_polynomial_size) as f64;

        let ell = params.br_decomposition_parameter.level as f64;

        let number_of_ct = 1.;

        let number_of_operations = number_of_ct * algorithmic_complexity_pbs(n, k, N, ell);

        let size = std::mem::size_of::<u64>() as f64;

        let pbs_cost = w4 * number_of_operations / (self.ncores as f64 * occupancy)
            + (w1 * n * (2. + ell * N * square(k + 1.))
                + 2. * N * ell * (w2 + w3 * square(k + 1.)))
                * size;

        pbs_cost
    }

    #[allow(clippy::let_and_return)]
    fn ks_complexity(
        &self,
        params: KeyswitchParameters,
        ciphertext_modulus_log: u32,
    ) -> Complexity {
        let GpuKsComplexity {
            w1,
            w2,
            w3,
            w4,
            occupancy,
            number_of_sm,
        } = self.ks;

        let na = params.input_lwe_dimension.0 as f64;

        let nb = params.output_lwe_dimension.0 as f64;

        let ell = params.ks_decomposition_parameter.level as f64;

        let number_of_ct = 1.;

        let number_of_operations =
            number_of_ct * algorithmic_complexity_ks(na, nb, ell, ciphertext_modulus_log as f64);

        let size = std::mem::size_of::<u64>() as f64;

        let ks_cost = w4 * number_of_operations / (self.ncores as f64 * occupancy)
            + w1 * (number_of_ct * ((na + 1.) + (nb + 1.)) + ell * (nb + 1.) * na) * size
            + w2 * number_of_ct * nb * size
            + w3 * (number_of_ct / number_of_ct.min(number_of_sm as f64 * 12.)).ceil()
                * ((na + 1.) + (nb + 1.))
            + ell * (nb + 1.) * size;

        ks_cost
    }

    fn levelled_complexity(
        &self,
        _sum_size: u64,
        _lwe_dimension: LweDimension,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        0.
    }
}

#[allow(non_snake_case)]
fn algorithmic_complexity_pbs(n: f64, k: f64, N: f64, ell: f64) -> f64 {
    n * (ell * (k + 1.) * N * (N.log2() + 1.)
        + (k + 1.) * N * (N.log2() + 1.)
        + N * ell * square(k + 1.))
}

#[allow(non_snake_case)]
fn algorithmic_complexity_ks(na: f64, nb: f64, ell: f64, log2_q: f64) -> f64 {
    na * nb * ell * log2_q
}
