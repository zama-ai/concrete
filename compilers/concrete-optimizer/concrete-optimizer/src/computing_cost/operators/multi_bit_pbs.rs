use super::super::complexity::Complexity;
use super::cmux;
use crate::parameters::PbsParameters;
use crate::utils::square;

#[derive(Default, Clone)]
pub struct MultiBitPbsComplexity {
    pub cmux: cmux::SimpleWithFactors,
}

impl MultiBitPbsComplexity {
    pub fn complexity(
        &self,
        params: PbsParameters,
        ciphertext_modulus_log: u32,
        grouping_factor: u32,
        jit_fft: bool,
    ) -> Complexity {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L163
        // grouping_factor: nb of sk bit bundled together
        let square_glwe_size = square(params.output_glwe_params.glwe_dimension as f64 + 1.);
        let cmux_cost = self
            .cmux
            .complexity(params.cmux_parameters(), ciphertext_modulus_log);

        let ggsw_size = params.br_decomposition_parameter.level as f64
            * square_glwe_size
            * params.output_glwe_params.polynomial_size() as f64;

        // JIT fourier transform for the GGSW
        let jit_fft_complexity = if jit_fft {
            ggsw_size * params.output_glwe_params.log2_polynomial_size as f64
        } else {
            0.
        };

        (params.internal_lwe_dimension.0 as f64) / (grouping_factor as f64) * cmux_cost
            + 2. * (f64::exp2(grouping_factor as f64) - 1.) * ggsw_size
            + jit_fft_complexity
    }
}
