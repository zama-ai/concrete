use super::super::complexity::Complexity;
use super::cmux;
use crate::parameters::PbsParameters;

pub trait PbsComplexity {
    fn complexity(
        &self,
        params: PbsParameters,
        ciphertext_modulus_log: u64, // log2_q
    ) -> Complexity;
}

pub struct CmuxProportional<CMUX: cmux::CmuxComplexity> {
    cmux: CMUX,
}

impl<CMUX: cmux::CmuxComplexity> PbsComplexity for CmuxProportional<CMUX> {
    fn complexity(&self, params: PbsParameters, ciphertext_modulus_log: u64) -> Complexity {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L163
        let cmux_cost = self
            .cmux
            .complexity(params.cmux_parameters(), ciphertext_modulus_log);
        (params.internal_lwe_dimension.0 as f64) * cmux_cost
    }
}

pub type Default = CmuxProportional<cmux::Default>;

pub const DEFAULT: Default = CmuxProportional {
    cmux: cmux::DEFAULT,
};

#[cfg(test)]
pub mod tests {

    use crate::computing_cost::operators::pbs::PbsParameters;
    use crate::parameters::{BrDecompositionParameters, GlweParameters, LweDimension};

    use super::super::cmux;
    use super::{CmuxProportional, Default, PbsComplexity, DEFAULT};

    pub const COST_AWS: Default = CmuxProportional {
        cmux: cmux::tests::COST_AWS,
    };

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

        let actual = DEFAULT.complexity(pbs_param1, 32);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 249957554585600.0;
        let actual = DEFAULT.complexity(pbs_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 208532086206064.16;
        let actual = COST_AWS.complexity(pbs_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
