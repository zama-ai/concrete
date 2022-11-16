use std::sync::Arc;

use crate::computing_cost::complexity_model::ComplexityModel;
use crate::computing_cost::cpu::CpuComplexity;
use crate::computing_cost::gpu::GpuComplexity;

#[derive(Clone, Copy)]
pub enum ProcessingUnit {
    Cpu,
    Gpu {
        pbs_type: GpuPbsType,
        number_of_sm: u64,
    },
}

#[derive(Clone, Copy)]
pub enum GpuPbsType {
    Lowlat,
    Amortized,
}

impl ProcessingUnit {
    pub fn ks_to_string(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu { .. } => "gpu",
        }
    }
    pub fn br_to_string(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                ..
            } => "gpu_lowlat",
            Self::Gpu {
                pbs_type: GpuPbsType::Amortized,
                ..
            } => "gpu_amortized",
        }
    }
    pub fn complexity_model(self) -> Arc<dyn ComplexityModel> {
        match self {
            Self::Cpu => Arc::new(CpuComplexity::default()),
            Self::Gpu {
                pbs_type: GpuPbsType::Amortized,
                number_of_sm,
            } => Arc::new(GpuComplexity::default_amortized_u64(number_of_sm)),
            Self::Gpu {
                pbs_type: GpuPbsType::Lowlat,
                number_of_sm,
            } => Arc::new(GpuComplexity::default_lowlat_u64(number_of_sm)),
        }
    }
}
