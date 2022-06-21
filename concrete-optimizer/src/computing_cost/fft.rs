use super::complexity::Complexity;

pub trait FftComplexity {
    fn fft_complexity(&self, size: f64) -> Complexity;
    fn ifft_complexity(&self, size: f64) -> Complexity;
}

/** Standard fft complexity model */
pub struct AsymptoticWithFactors {
    factor_fft: f64,  // factor applied on asymptotic complexity
    factor_ifft: f64, // factor applied on asymptotic complexity
}

impl FftComplexity for AsymptoticWithFactors {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L109

    #[inline] // forces to share log2 computation
    fn fft_complexity(&self, size: f64) -> Complexity {
        size * size.log2() * self.factor_fft
    }

    #[inline] // forces to share log2 computation
    fn ifft_complexity(&self, size: f64) -> Complexity {
        size * size.log2() * self.factor_ifft
    }
}

/** Standard fft complexity with 1.0 factor*/
pub const DEFAULT: AsymptoticWithFactors = AsymptoticWithFactors {
    factor_fft: 1.0,
    factor_ifft: 1.0,
};

#[cfg(test)]
pub mod tests {
    use crate::computing_cost::fft;
    use crate::computing_cost::fft::{AsymptoticWithFactors, FftComplexity};

    /** Standard fft complexity with X factors*/
    pub const COST_AWS: AsymptoticWithFactors = AsymptoticWithFactors {
        // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L150
        factor_fft: 0.202_926_951_153_089_17,
        factor_ifft: 0.407_795_078_512_891,
    };

    #[test]
    fn golden_python_prototype() {
        let golden_fft = 664.385_618_977_472_4;
        let actual_fft = fft::DEFAULT.fft_complexity(100.);
        approx::assert_relative_eq!(golden_fft, actual_fft, epsilon = f64::EPSILON);

        let golden_ifft = 664.385_618_977_472_4;
        let actual_ifft = fft::DEFAULT.ifft_complexity(100.);
        approx::assert_relative_eq!(golden_ifft, actual_ifft, epsilon = f64::EPSILON);
    }
}
