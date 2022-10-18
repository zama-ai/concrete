use crate::computing_cost::complexity_model::ComplexityModel;
use crate::config;
use crate::config::GpuPbsType;
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

impl SearchSpace {
    pub fn default_cpu() -> Self {
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

    pub fn default_gpu_lowlat() -> Self {
        // https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_low_latency.cu#L156
        let glwe_log_polynomial_sizes: Vec<u64> = (9..=11).collect();

        // https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_low_latency.cu#L154
        let glwe_dimensions: Vec<u64> = vec![1];

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();

        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        }
    }

    pub fn default_gpu_amortized() -> Self {
        // https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_amortized.cu#L79
        let glwe_log_polynomial_sizes: Vec<u64> = (9..=13).collect();

        // https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_amortized.cu#L78
        let glwe_dimensions: Vec<u64> = vec![1];

        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();

        Self {
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        }
    }
    pub fn default(processing_unit: config::ProcessingUnit) -> Self {
        match processing_unit {
            config::ProcessingUnit::Cpu => Self::default_cpu(),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Amortized,
                ..
            } => Self::default_gpu_amortized(),
            config::ProcessingUnit::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                ..
            } => Self::default_gpu_lowlat(),
        }
    }
}

// https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_amortized.cu#L77
// https://github.com/zama-ai/concrete-core/blob/6b52182ab44c4b39ddebca1c457e1096fb687801/concrete-cuda/cuda/src/bootstrap_low_latency.cu#L153
pub const MAX_LOG2_BASE_GPU: u64 = 16;

pub const MAX_LOG2_BASE_CPU: u64 = 64;
