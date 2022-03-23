use super::super::complexity::Complexity;
use super::super::fft;
use fft::FftComplexity;

pub trait CmuxComplexity {
    #[allow(non_snake_case)]
    fn complexity(
        &self,
        glwe_polynomial_size: u64,         //N
        glwe_dimension: u64,               //k
        br_decomposition_level_count: u64, //l(BR)
        br_decomposition_base_log: u64,    //b(BR)
        ciphertext_modulus_log: u64,       //log2_q
    ) -> Complexity;
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

impl<FFT: FftComplexity> CmuxComplexity for SimpleWithFactors<FFT> {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L145
    #[allow(non_snake_case)]
    fn complexity(
        &self,
        glwe_polynomial_size: u64,         //N
        glwe_dimension: u64,               //k
        br_decomposition_level_count: u64, //l(BR)
        _br_decomposition_base_log: u64,   //b(BR)
        ciphertext_modulus_log: u64,       //log2_q
    ) -> Complexity {
        let f_glwe_polynomial_size = glwe_polynomial_size as f64;
        let f_glwe_size = (glwe_dimension + 1) as f64;
        let br_decomposition_level_count = br_decomposition_level_count as f64;
        let f_square_glwe_size = f_glwe_size * f_glwe_size;

        let additional_linear_fft_factor =
            final_additional_linear_fft_factor(self.linear_fft_factor, ciphertext_modulus_log);
        let additional_linear_ifft_factor =
            final_additional_linear_fft_factor(self.linear_ifft_factor, ciphertext_modulus_log);

        let fft_cost = f_glwe_size
            * br_decomposition_level_count
            * (self.fft.fft_complexity(glwe_polynomial_size)
                + additional_linear_fft_factor * f_glwe_polynomial_size);
        let ifft_cost = f_glwe_size
            * (self.fft.ifft_complexity(glwe_polynomial_size)
                + additional_linear_ifft_factor * f_glwe_polynomial_size);
        let br_cost = self.blind_rotate_factor
            * f_glwe_polynomial_size
            * br_decomposition_level_count
            * f_square_glwe_size;

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
    use super::*;

    pub const COST_AWS: Default = SimpleWithFactors {
        /* https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L145 */
        fft: fft::tests::COST_AWS,
        linear_fft_factor: Some(0.011647955063264166),
        linear_ifft_factor: Some(0.018836852582634938),
        blind_rotate_factor: 0.8418306429189878,
        constant_cost: 923.7542202718637,
    };

    #[test]
    fn golden_python_prototype() {
        let ignored = 0;
        let golden = 8.0;
        let actual = DEFAULT.complexity(1, 1, 1, ignored, 0);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 138621.0;
        let actual = DEFAULT.complexity(1, 20, 300, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 927.1215428435396;
        let actual = COST_AWS.complexity(1, 1, 1, ignored, 0);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 117019.72048983313;
        let actual = COST_AWS.complexity(1, 20, 300, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);

        let golden = 7651844.24194206;
        let actual = COST_AWS.complexity(1024, 10, 56, ignored, 64);
        approx::assert_relative_eq!(golden, actual, epsilon = f64::EPSILON);
    }
}
