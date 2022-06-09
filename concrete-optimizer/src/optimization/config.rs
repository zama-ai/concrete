use crate::computing_cost::complexity_model::ComplexityModel;
use crate::global_parameters::DEFAUT_DOMAINS;

#[derive(Clone, Copy, Debug)]
pub struct NoiseBoundConfig {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub ciphertext_modulus_log: u32,
}

#[derive(Clone, Copy)]
pub struct Config<'a> {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub ciphertext_modulus_log: u32,
    pub complexity_model: &'a dyn ComplexityModel,
}

#[derive(Clone, Debug)]
pub struct SearchSpace {
    pub glwe_log_polynomial_sizes: Vec<u64>,
    pub glwe_dimensions: Vec<u64>,
    pub internal_lwe_dimensions: Vec<u64>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();

        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        }
    }
}
