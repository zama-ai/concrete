use super::super::complexity::Complexity;
use super::cmux;

pub trait PbsComplexity {
    fn complexity(
        &self,
        lwe_dimension: u64,                //n
        glwe_polynomial_size: u64,         //N
        glwe_dimension: u64,               //k
        br_decomposition_level_count: u64, //l(BR)
        br_decomposition_base_log: u64,    //b(BR)
        ciphertext_modulus_log: u64,       // log2_q
    ) -> Complexity;
}

pub struct CmuxProportional<CMUX: cmux::CmuxComplexity> {
    cmux: CMUX,
}

impl<CMUX: cmux::CmuxComplexity> PbsComplexity for CmuxProportional<CMUX> {
    fn complexity(
        &self,
        lwe_dimension: u64,                //n
        glwe_polynomial_size: u64,         //N
        glwe_dimension: u64,               //k
        br_decomposition_level_count: u64, //l(BR)
        br_decomposition_base_log: u64,    //b(BR)
        ciphertext_modulus_log: u64,       //log2_q
    ) -> Complexity {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L163
        let cmux_cost = self.cmux.complexity(
            glwe_polynomial_size,
            glwe_dimension,
            br_decomposition_level_count,
            br_decomposition_base_log,
            ciphertext_modulus_log,
        );
        (lwe_dimension as f64) * cmux_cost
    }
}

pub type Default = CmuxProportional<cmux::Default>;

pub const DEFAULT: Default = CmuxProportional {
    cmux: cmux::DEFAULT,
};

#[cfg(test)]
pub mod tests {
    use super::super::cmux;
    use super::{CmuxProportional, Default, PbsComplexity, DEFAULT};

    pub const COST_AWS: Default = CmuxProportional {
        cmux: cmux::tests::COST_AWS,
    };

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 8.0;
        let actual = DEFAULT.complexity(1, 1, 1, 1, ignored, 32);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 249957554585600.0;
        let actual = DEFAULT.complexity(1024, 4096, 1024, 56, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 208532086206064.16;
        let actual = COST_AWS.complexity(1024, 4096, 1024, 56, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
