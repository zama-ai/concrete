use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::config::GpuPbsType;
use crate::global_parameters::{Range, DEFAUT_DOMAINS};
use crate::parameters::GlweParameters;

#[derive(Clone, Copy, Debug)]
pub struct NoiseBoundConfig {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub ciphertext_modulus_log: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum PublicKey {
    None,
    Classic,
    Compact,
}

impl PublicKey {
    pub fn filter_lwe_dimension(self, lwedim: u64) -> bool {
        match self {
            Self::None | Self::Classic => true,
            Self::Compact => {
                // Test if the lwe dimension is a power of 2
                (lwedim & (lwedim - 1)) == 0
            }
        }
    }
    pub fn filter_glwe_params(self, glwe_params: &GlweParameters) -> bool {
        match self {
            Self::None | Self::Classic => true,
            Self::Compact => self.filter_lwe_dimension(
                glwe_params.glwe_dimension * (2 << glwe_params.log2_polynomial_size),
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Config<'a> {
    pub security_level: u64,
    pub maximum_acceptable_error_probability: f64,
    pub key_sharing: bool,
    pub ciphertext_modulus_log: u32,
    pub fft_precision: u32,
    pub complexity_model: &'a dyn ComplexityModel,
    pub composable: bool,
    pub public_keys: PublicKey,
}

impl Config<'_> {
    pub fn minimal_variance_lwe(self, lwe_dimension: u64) -> f64 {
        GlweParameters {
            glwe_dimension: lwe_dimension,
            log2_polynomial_size: 0,
        }
        .minimal_variance_for_public_key(self.ciphertext_modulus_log, self.security_level)
    }
}

#[derive(Clone, Debug)]
pub struct SearchSpace {
    glwe_log_polynomial_sizes: Vec<u64>,
    glwe_dimensions: Vec<u64>,
    pub internal_lwe_dimensions: Vec<u64>,
    pub levelled_only_lwe_dimensions: Range,
    public_key: PublicKey,
}

impl SearchSpace {
    pub fn new(
        glwe_log_polynomial_sizes: Vec<u64>,
        glwe_dimensions: Vec<u64>,
        internal_lwe_dimensions: Vec<u64>,
        levelled_only_lwe_dimensions: Range,
        public_key: PublicKey,
    ) -> Self {
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
            public_key,
        }
    }
    pub fn cpu_with_glwe_dimensions(public_key: PublicKey, glwe_dimensions: Vec<u64>) -> Self {
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
            public_key,
        }
    }
    pub fn default_cpu(public_key: PublicKey) -> Self {
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        Self::cpu_with_glwe_dimensions(public_key, glwe_dimensions)
    }

    pub fn default_gpu_lowlat(public_key: PublicKey) -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_low_latency.cu
        let glwe_log_polynomial_sizes: Vec<u64> = (8..=14).collect();

        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
            public_key,
        }
    }

    pub fn default_gpu_amortized(public_key: PublicKey) -> Self {
        // See backends/concrete_cuda/implementation/src/bootstrap_amortized.cu
        let glwe_log_polynomial_sizes: Vec<u64> = (8..=14).collect();

        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let levelled_only_lwe_dimensions = DEFAUT_DOMAINS.free_lwe;
        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
            levelled_only_lwe_dimensions,
            public_key,
        }
    }
    pub fn default(processing_unit: config::ProcessingUnit, public_key: PublicKey) -> Self {
        match processing_unit {
            config::ProcessingUnit::Cpu => Self::default_cpu(public_key),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Amortized,
                ..
            } => Self::default_gpu_amortized(public_key),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                ..
            } => Self::default_gpu_lowlat(public_key),
        }
    }

    pub fn get_glwe_params(self) -> impl Iterator<Item = GlweParameters> {
        self.glwe_dimensions
            .clone()
            .into_iter()
            .flat_map(move |glwe_dimension| {
                self.glwe_log_polynomial_sizes.clone().into_iter().map(
                    move |log2_polynomial_size| GlweParameters {
                        log2_polynomial_size,
                        glwe_dimension,
                    },
                )
            })
            .filter(move |glwe_param| self.public_key.filter_glwe_params(glwe_param))
    }

    pub fn get_levelled_only_lwe_dimensions(self) -> impl Iterator<Item = u64> {
        self.levelled_only_lwe_dimensions
            .clone()
            .into_iter()
            .filter(move |lwedim| self.public_key.filter_lwe_dimension(*lwedim))
    }
}
