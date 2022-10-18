use super::super::complexity::Complexity;
use super::super::fft;
use crate::parameters::CmuxParameters;
use crate::utils::square;

#[derive(Clone)]
pub struct SimpleWithFactors {
    fft: fft::AsymptoticWithFactors,
    blind_rotate_factor: f64,
    constant_cost: f64, // global const
}

impl SimpleWithFactors {
    pub fn fft_complexity(
        &self,
        glwe_polynomial_size: f64,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        self.fft.fft_complexity(glwe_polynomial_size) + glwe_polynomial_size
    }

    fn ifft_complexity(
        &self,
        glwe_polynomial_size: f64,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        self.fft.ifft_complexity(glwe_polynomial_size) + glwe_polynomial_size
    }

    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L145
    pub fn complexity(&self, params: CmuxParameters, ciphertext_modulus_log: u32) -> Complexity {
        let glwe_polynomial_size = params.output_glwe_params.polynomial_size() as f64;

        let f_glwe_size = (params.output_glwe_params.glwe_dimension + 1) as f64;
        let br_decomposition_level_count = params.br_decomposition_parameter.level as f64;

        let fft_cost = f_glwe_size
            * br_decomposition_level_count
            * self.fft_complexity(glwe_polynomial_size, ciphertext_modulus_log);

        let ifft_cost =
            f_glwe_size * self.ifft_complexity(glwe_polynomial_size, ciphertext_modulus_log);

        let br_cost = square(f_glwe_size)
            * br_decomposition_level_count
            * glwe_polynomial_size
            * self.blind_rotate_factor;

        fft_cost + ifft_cost + br_cost + self.constant_cost
    }
}

impl Default for SimpleWithFactors {
    fn default() -> Self {
        Self {
            fft: fft::AsymptoticWithFactors::default(),
            blind_rotate_factor: 1.0,
            constant_cost: 0.0,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::parameters::{BrDecompositionParameters, GlweParameters};

    use super::*;

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 8.0;

        let cmux_param1 = CmuxParameters {
            br_decomposition_parameter: BrDecompositionParameters {
                level: 1,
                log2_base: ignored,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: 0,
                glwe_dimension: 1,
            },
        };

        let cmux_param2 = CmuxParameters {
            br_decomposition_parameter: BrDecompositionParameters {
                level: 300,
                log2_base: ignored,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: 0,
                glwe_dimension: 20,
            },
        };

        let comp = SimpleWithFactors::default();

        let actual = comp.complexity(cmux_param1, 0);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 138_621.0;
        let actual = comp.complexity(cmux_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
