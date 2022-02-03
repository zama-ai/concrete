use super::super::complexity::Complexity;

pub trait KeySwitchLWEComplexity {
    fn complexity(
        &self,
        input_lwe_dimension: u64,       //n_big
        output_lwe_dimension: u64,      //n_small
        decomposition_level_count: u64, //l(BS)
        decomposition_base_log: u64,    //b(BS)
        ciphertext_modulus_log: u64,    //log2_q
    ) -> Complexity;
}

pub struct Default;

impl KeySwitchLWEComplexity for Default {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L91
    fn complexity(
        &self,
        input_lwe_dimension: u64,       //n_big
        output_lwe_dimension: u64,      //n_small
        decomposition_level_count: u64, //l(KS)
        _decomposition_base_log: u64,   //b(KS)
        _ciphertext_modulus_log: u64,   //log2_q
    ) -> Complexity {
        let output_lwe_size = output_lwe_dimension + 1;
        let count_decomposition = input_lwe_dimension * decomposition_level_count;
        let count_mul = input_lwe_dimension * decomposition_level_count * output_lwe_size;
        let count_add = (input_lwe_dimension * decomposition_level_count - 1) * output_lwe_size + 1;
        (count_decomposition + count_mul + count_add) as Complexity
    }
}

pub struct SimpleProductWithFactor {
    factor: f64,
}

impl KeySwitchLWEComplexity for SimpleProductWithFactor {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L100
    fn complexity(
        &self,
        input_lwe_dimension: u64,       //n_big
        output_lwe_dimension: u64,      //n_small
        decomposition_level_count: u64, //l(BS)
        _decomposition_base_log: u64,   //b(BS)
        ciphertext_modulus_log: u64,    //log2_q
    ) -> Complexity {
        let product = input_lwe_dimension
            * output_lwe_dimension
            * decomposition_level_count
            * ciphertext_modulus_log;
        self.factor * (product as f64)
    }
}

pub const DEFAULT: Default = Default;

#[cfg(test)]
mod tests {
    use super::*;
    pub const COST_AWS: SimpleProductWithFactor = SimpleProductWithFactor {
        factor: 0.12547239853890443,
    };

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 134313984.0;
        let actual = DEFAULT.complexity(1024, 2048, 32, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 538899848.2752727;
        let actual = COST_AWS.complexity(1024, 2048, 32, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
