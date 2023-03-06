use super::complexity::Complexity;

/** Standard fft complexity model */
#[derive(Clone)]
pub struct AsymptoticWithFactors {
    factor_fft: f64,  // factor applied on asymptotic complexity
    factor_ifft: f64, // factor applied on asymptotic complexity
}

impl AsymptoticWithFactors {
    // https://github.com/zama-ai/concrete-optimizer/blob/prototype/python/optimizer/noise_formulas/bootstrap.py#L109

    #[inline] // forces to share log2 computation
    pub fn fft_complexity(&self, size: f64) -> Complexity {
        size * size.log2() * self.factor_fft
    }

    #[inline] // forces to share log2 computation
    pub fn ifft_complexity(&self, size: f64) -> Complexity {
        size * size.log2() * self.factor_ifft
    }
}

impl Default for AsymptoticWithFactors {
    fn default() -> Self {
        Self {
            factor_fft: 1.0,
            factor_ifft: 1.0,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::computing_cost::fft;

    #[test]
    fn golden_python_prototype() {
        let golden_fft = 664.385_618_977_472_4;
        let actual_fft = fft::AsymptoticWithFactors::default().fft_complexity(100.);
        approx::assert_relative_eq!(golden_fft, actual_fft, epsilon = f64::EPSILON);

        let golden_ifft = 664.385_618_977_472_4;
        let actual_ifft = fft::AsymptoticWithFactors::default().ifft_complexity(100.);
        approx::assert_relative_eq!(golden_ifft, actual_ifft, epsilon = f64::EPSILON);
    }
}
