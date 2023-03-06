use super::super::complexity::Complexity;
use super::cmux;
use crate::parameters::PbsParameters;

#[derive(Default, Clone)]
pub struct PbsComplexity {
    pub cmux: cmux::SimpleWithFactors,
}

impl PbsComplexity {
    pub fn complexity(&self, params: PbsParameters, ciphertext_modulus_log: u32) -> Complexity {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L163

        let cmux_cost = self
            .cmux
            .complexity(params.cmux_parameters(), ciphertext_modulus_log);
        (params.internal_lwe_dimension.0 as f64) * cmux_cost
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::computing_cost::operators::pbs::PbsParameters;
    use crate::parameters::{BrDecompositionParameters, GlweParameters, LweDimension};

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 8.0;

        let pbs_param1 = PbsParameters {
            internal_lwe_dimension: LweDimension(1),
            br_decomposition_parameter: BrDecompositionParameters {
                level: 1,
                log2_base: ignored,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: 0,
                glwe_dimension: 1,
            },
        };

        let pbs_param2 = PbsParameters {
            internal_lwe_dimension: LweDimension(1024),
            br_decomposition_parameter: BrDecompositionParameters {
                level: 56,
                log2_base: ignored,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: 12,
                glwe_dimension: 1024,
            },
        };

        let complexity = PbsComplexity::default();

        let actual = complexity.complexity(pbs_param1, 32);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 249_957_554_585_600.0;
        let actual = complexity.complexity(pbs_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
