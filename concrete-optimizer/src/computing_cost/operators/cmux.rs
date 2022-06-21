use super::super::complexity::Complexity;
use super::super::fft;
use crate::parameters::CmuxParameters;
use crate::utils::square;
use fft::FftComplexity;

pub trait CmuxComplexity {
    #[allow(non_snake_case)]
    fn complexity(&self, params: CmuxParameters, ciphertext_modulus_log: u64) -> Complexity;
}

#[allow(non_snake_case)]
pub struct SimpleWithFactors<FFT: FftComplexity> {
    fft: FFT,
    linear_fft_factor: Option<f64>, // fft additional linear factor cost, none => size | some(w) => size * w * log2_q
    linear_ifft_factor: Option<f64>, // ifft additional linear factor cost
    blind_rotate_factor: f64,
    constant_cost: f64, // global const
}

fn final_additional_linear_fft_factor(factor: Option<f64>, integer_size: u64) -> f64 {
    match factor {
        Some(w) => w * (integer_size as f64),
        None => 1.0,
    }
}

impl<FFT: FftComplexity> SimpleWithFactors<FFT> {
    pub fn fft_complexity(
        &self,
        glwe_polynomial_size: f64,
        ciphertext_modulus_log: u64,
    ) -> Complexity {
        let additional_linear_fft_factor =
            final_additional_linear_fft_factor(self.linear_fft_factor, ciphertext_modulus_log);
        self.fft.fft_complexity(glwe_polynomial_size)
            + additional_linear_fft_factor * glwe_polynomial_size
    }

    fn ifft_complexity(
        &self,
        glwe_polynomial_size: f64,
        ciphertext_modulus_log: u64,
    ) -> Complexity {
        let additional_linear_ifft_factor =
            final_additional_linear_fft_factor(self.linear_ifft_factor, ciphertext_modulus_log);

        self.fft.ifft_complexity(glwe_polynomial_size)
            + additional_linear_ifft_factor * glwe_polynomial_size
    }
}

impl<FFT: FftComplexity> CmuxComplexity for SimpleWithFactors<FFT> {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L145
    #[allow(non_snake_case)]
    fn complexity(&self, params: CmuxParameters, ciphertext_modulus_log: u64) -> Complexity {
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

pub type Default = SimpleWithFactors<fft::AsymptoticWithFactors>;

pub const DEFAULT: Default = SimpleWithFactors {
    fft: fft::DEFAULT,
    linear_fft_factor: None,
    linear_ifft_factor: None,
    blind_rotate_factor: 1.0,
    constant_cost: 0.0,
};

#[cfg(test)]
pub mod tests {
    use crate::parameters::{BrDecompositionParameters, GlweParameters};

    use super::*;

    pub const COST_AWS: Default = SimpleWithFactors {
        /* https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L145 */
        fft: fft::tests::COST_AWS,
        linear_fft_factor: Some(0.011_647_955_063_264_166),
        linear_ifft_factor: Some(0.018_836_852_582_634_938),
        blind_rotate_factor: 0.841_830_642_918_987_8,
        constant_cost: 923.754_220_271_863_7,
    };

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

        let cmux_param3 = CmuxParameters {
            br_decomposition_parameter: BrDecompositionParameters {
                level: 56,
                log2_base: ignored,
            },
            output_glwe_params: GlweParameters {
                log2_polynomial_size: 10,
                glwe_dimension: 10,
            },
        };

        let actual = DEFAULT.complexity(cmux_param1, 0);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 138_621.0;
        let actual = DEFAULT.complexity(cmux_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 927.121_542_843_539_6;
        let actual = COST_AWS.complexity(cmux_param1, 0);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 117_019.720_489_833_13;
        let actual = COST_AWS.complexity(cmux_param2, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 7_651_844.241_942_06;
        let actual = COST_AWS.complexity(cmux_param3, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
