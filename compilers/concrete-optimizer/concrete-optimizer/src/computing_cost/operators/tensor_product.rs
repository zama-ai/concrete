use super::super::complexity::Complexity;
use super::super::fft;
use crate::parameters::TensorProductGlweParameters;

#[derive(Default, Clone)]
pub struct TensorProductGlweComplexity {
    fft: fft::AsymptoticWithFactors,
}

impl TensorProductGlweComplexity {
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

    pub fn complexity(
        &self,
        params: TensorProductGlweParameters,
        ciphertext_modulus_log: u32,
    ) -> Complexity {
        let glwe_polynomial_size =
            2_f64.powi(params.output_glwe_params.log2_polynomial_size as i32);

        let k_plus = ((params.input_glwe_params.glwe_dimension + 1)
            * (params.input_glwe_params.glwe_dimension + 2)) as f64
            / 2.;
        let k_star = (params.input_glwe_params.glwe_dimension
            * (params.input_glwe_params.glwe_dimension + 1)) as f64
            / 2.;

        //C_{multFFT}/C_{addFFT} complexity of adding/multiplying two numbers assumed to be 1
        //C_{dec} complexity of decomposing a number assumed to be 1 in accordance with key switching
        let complexity_tensor_product = 2.
            * (params.input_glwe_params.glwe_dimension as f64 + 1.)
            * self.fft_complexity(glwe_polynomial_size, ciphertext_modulus_log)
            + k_plus * self.ifft_complexity(glwe_polynomial_size, ciphertext_modulus_log)
            + (params.input_glwe_params.glwe_dimension as f64 + 1.).powi(2) * glwe_polynomial_size
            + k_star * glwe_polynomial_size;

        let complexity_relin =
            glwe_polynomial_size * params.ks_decomposition_parameters.level as f64 * k_star
                + k_star
                    * params.ks_decomposition_parameters.level as f64
                    * self.fft_complexity(glwe_polynomial_size, ciphertext_modulus_log)
                + k_star
                    * params.ks_decomposition_parameters.level as f64
                    * (params.input_glwe_params.glwe_dimension as f64 + 1.)
                    * glwe_polynomial_size as f64
                + (k_star * params.ks_decomposition_parameters.level as f64 - 1.)
                    * (params.input_glwe_params.glwe_dimension as f64 + 1.)
                    * glwe_polynomial_size as f64
                + (params.input_glwe_params.glwe_dimension as f64 + 1.)
                    * self.ifft_complexity(glwe_polynomial_size, ciphertext_modulus_log);

        (complexity_tensor_product + complexity_relin) as Complexity
    }
}
