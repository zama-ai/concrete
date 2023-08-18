use super::super::complexity::Complexity;
use super::super::fft;
use crate::parameters::KeyswitchGlweParameters;

#[derive(Clone)]
pub struct SimpleWithFactors {
    fft: fft::AsymptoticWithFactors,
    glwe_keyswitch_factor: f64,
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

    /*pub fn complexity(
        &self,
        params: KeyswitchGlweParameters,
        _ciphertext_modulus_log: u32,
    ) -> Complexity {
        let input_glwe_dimension = params.input_glwe_params.glwe_dimension as f64;
        let output_glwe_dimension = params.output_glwe_params.glwe_dimension as f64;
        let level = params.ks_decomposition_parameter.level as f64;
        let poly_dim = 2_f64.powi(params.input_glwe_params.log2_polynomial_size as i32);

        let output_glwe_size = output_glwe_dimension + 1.;
        let count_decomposition = input_glwe_dimension * level * poly_dim;
        let count_mul = input_glwe_dimension * level * output_glwe_size * poly_dim;
        let count_add = (input_glwe_dimension * level * poly_dim - 1.) * output_glwe_size + 1.;
        (count_decomposition + count_mul + count_add) as Complexity
    }*/
    pub fn complexity(&self, params: KeyswitchGlweParameters, ciphertext_modulus_log: u32) -> Complexity {
        let glwe_polynomial_size = params.output_glwe_params.polynomial_size() as f64;

        let input_glwe_dimension = params.input_glwe_params.glwe_dimension as f64;
        let output_glwe_size = (params.output_glwe_params.glwe_dimension + 1) as f64;
        let ks_decomposition_level_count = params.ks_decomposition_parameter.level as f64;

        let fft_cost = input_glwe_dimension
            * ks_decomposition_level_count
            * self.fft_complexity(glwe_polynomial_size, ciphertext_modulus_log);

        let ifft_cost =
            output_glwe_size * self.ifft_complexity(glwe_polynomial_size, ciphertext_modulus_log);

        let mult_cost_in_FFT_domain = input_glwe_dimension
            * ks_decomposition_level_count
            * output_glwe_size
            * glwe_polynomial_size
            * self.glwe_keyswitch_factor;

        fft_cost + ifft_cost + mult_cost_in_FFT_domain + self.constant_cost
    }
}

impl Default for SimpleWithFactors {
    fn default() -> Self {
        Self {
            fft: fft::AsymptoticWithFactors::default(),
            glwe_keyswitch_factor: 1.0,
            constant_cost: 0.0,
        }
    }
}
