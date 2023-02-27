use super::super::complexity::Complexity;
use crate::parameters::KeyswitchParameters;

#[derive(Clone)]
pub struct KsComplexity;

impl KsComplexity {
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::unused_self)]
    pub fn complexity(
        &self,
        params: KeyswitchParameters,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/keyswitch.py#L91
        let input_lwe_dimension = params.input_lwe_dimension.0 as i64;
        let output_lwe_dimension = params.output_lwe_dimension.0 as i64;
        let level = params.ks_decomposition_parameter.level as i64;

        let output_lwe_size = output_lwe_dimension + 1;
        let count_decomposition = input_lwe_dimension * level;
        let count_mul = input_lwe_dimension * level * output_lwe_size;
        let count_add = (input_lwe_dimension * level - 1) * output_lwe_size + 1;
        (count_decomposition + count_mul + count_add) as Complexity
    }
}

#[cfg(test)]
mod tests {
    use crate::parameters::{KsDecompositionParameters, LweDimension};

    use super::*;

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 134_313_984.0;

        let ks_params = KeyswitchParameters {
            input_lwe_dimension: LweDimension(1024),
            output_lwe_dimension: LweDimension(2048),
            ks_decomposition_parameter: KsDecompositionParameters {
                level: 32,
                log2_base: ignored,
            },
        };

        let actual = KsComplexity.complexity(ks_params, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
