use crate::parameters::KeyswitchParameters;

use super::super::complexity::Complexity;

pub trait KeySwitchLWEComplexity {
    fn complexity(&self, params: KeyswitchParameters, ciphertext_modulus_log: u64) -> Complexity;
}

pub struct Default;

impl KeySwitchLWEComplexity for Default {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L91
    fn complexity(&self, params: KeyswitchParameters, _ciphertext_modulus_log: u64) -> Complexity {
        let input_lwe_dimension = params.input_lwe_dimension.0;
        let output_lwe_dimension = params.output_lwe_dimension.0;
        let level = params.ks_decomposition_parameter.level;

        let output_lwe_size = output_lwe_dimension + 1;
        let count_decomposition = input_lwe_dimension * level;
        let count_mul = input_lwe_dimension * level * output_lwe_size;
        let count_add = (input_lwe_dimension * level - 1) * output_lwe_size + 1;
        (count_decomposition + count_mul + count_add) as Complexity
    }
}

pub struct SimpleProductWithFactor {
    factor: f64,
}

impl KeySwitchLWEComplexity for SimpleProductWithFactor {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L100
    fn complexity(&self, params: KeyswitchParameters, ciphertext_modulus_log: u64) -> Complexity {
        let product = params.input_lwe_dimension.0
            * params.output_lwe_dimension.0
            * params.ks_decomposition_parameter.level
            * ciphertext_modulus_log;
        self.factor * (product as f64)
    }
}

pub const DEFAULT: Default = Default;

#[cfg(test)]
mod tests {
    use crate::parameters::{KsDecompositionParameters, LweDimension};

    use super::*;
    pub const COST_AWS: SimpleProductWithFactor = SimpleProductWithFactor {
        factor: 0.12547239853890443,
    };

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 134313984.0;

        let ks_params = KeyswitchParameters {
            input_lwe_dimension: LweDimension(1024),
            output_lwe_dimension: LweDimension(2048),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: 32,
                log2_base: ignored,
            },
        };

        let actual = DEFAULT.complexity(ks_params, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 538899848.2752727;
        let actual = COST_AWS.complexity(ks_params, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
