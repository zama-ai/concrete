use super::super::complexity::Complexity;
//use super::super::fft;
use super::keyswitch_glwe::SimpleWithFactors;
use crate::parameters::{KeyswitchGlweParameters, TracePackingParameters};

#[derive(Default, Clone)]
pub struct TracePackingComplexity {
    //fft: fft::AsymptoticWithFactors,
    pub ks_glwe: SimpleWithFactors,
}

impl TracePackingComplexity {
    pub fn complexity(
        &self,
        params: TracePackingParameters,
        ciphertext_modulus_log: u32,
        index_set: &[usize],
    ) -> Complexity {
        let ks_glwe_params = KeyswitchGlweParameters {
            input_glwe_params: params.output_glwe_params,
            output_glwe_params: params.output_glwe_params,
            ks_decomposition_parameter: params.ks_decomposition_parameter,
        };

        let glwe_polynomial_size =
            2_f64.powi(params.output_glwe_params.log2_polynomial_size as i32);

        let ks_glwe_complexity = self
            .ks_glwe
            .complexity(ks_glwe_params, ciphertext_modulus_log);
        let mut complexity_tp = 0.;
        for l in 1..params.output_glwe_params.log2_polynomial_size as u32 {
            for i in 0..glwe_polynomial_size as usize / 2usize.pow(l) - 1 {
                if index_set.clone().into_iter().any(|el| *el != i)
                    && index_set
                        .clone()
                        .into_iter()
                        .any(|el| *el != glwe_polynomial_size as usize / 2usize.pow(l) + i)
                {
                } else {
                    complexity_tp = complexity_tp
                        + (4. * glwe_polynomial_size as f64 + 2. * ks_glwe_complexity as f64);
                }
            }
        }
        complexity_tp
    }
}
