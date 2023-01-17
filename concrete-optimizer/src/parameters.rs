pub use grouped::*;
pub use individual::*;
pub use range::*;

mod individual {
    use concrete_security_curves::gaussian::security::minimal_variance_glwe;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
    pub struct KsDecompositionParameters {
        pub level: u64,
        pub log2_base: u64,
    }

    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
    pub struct BrDecompositionParameters {
        pub level: u64,
        pub log2_base: u64,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
    pub struct GlweParameters {
        pub log2_polynomial_size: u64,
        pub glwe_dimension: u64,
    }

    impl GlweParameters {
        pub fn polynomial_size(self) -> u64 {
            1 << self.log2_polynomial_size
        }
        pub fn sample_extract_lwe_dimension(self) -> u64 {
            self.glwe_dimension << self.log2_polynomial_size
        }

        pub fn minimal_variance(self, ciphertext_modulus_log: u32, security_level: u64) -> f64 {
            minimal_variance_glwe(
                self.glwe_dimension,
                self.polynomial_size(),
                ciphertext_modulus_log,
                security_level,
            )
        }
    }

    #[derive(Clone, Copy)]
    pub struct LweDimension(pub u64);

    #[derive(Copy, Clone)]
    pub struct InputParameter {
        pub lwe_dimension: u64,
    }

    #[derive(Copy, Clone)]
    pub struct AtomicPatternParameters {
        pub input_lwe_dimension: LweDimension,
        pub ks_decomposition_parameter: KsDecompositionParameters,
        pub internal_lwe_dimension: LweDimension,
        pub br_decomposition_parameter: BrDecompositionParameters,
        pub output_glwe_params: GlweParameters,
    }

    impl AtomicPatternParameters {
        pub fn pbs_parameters(self) -> PbsParameters {
            PbsParameters {
                internal_lwe_dimension: self.internal_lwe_dimension,
                br_decomposition_parameter: self.br_decomposition_parameter,
                output_glwe_params: self.output_glwe_params,
            }
        }

        pub fn ks_parameters(self) -> KeyswitchParameters {
            KeyswitchParameters {
                input_lwe_dimension: self.input_lwe_dimension,
                output_lwe_dimension: self.internal_lwe_dimension,
                ks_decomposition_parameter: self.ks_decomposition_parameter,
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct PbsParameters {
        pub internal_lwe_dimension: LweDimension,
        pub br_decomposition_parameter: BrDecompositionParameters,
        pub output_glwe_params: GlweParameters,
    }

    impl PbsParameters {
        pub fn cmux_parameters(self) -> CmuxParameters {
            CmuxParameters {
                br_decomposition_parameter: self.br_decomposition_parameter,
                output_glwe_params: self.output_glwe_params,
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct CmuxParameters {
        pub br_decomposition_parameter: BrDecompositionParameters,
        pub output_glwe_params: GlweParameters,
    }

    #[derive(Clone, Copy)]
    pub struct KeyswitchParameters {
        pub input_lwe_dimension: LweDimension,  //n_big
        pub output_lwe_dimension: LweDimension, //n_small
        pub ks_decomposition_parameter: KsDecompositionParameters,
    }
}

mod range {

    use crate::global_parameters::Range;

    #[derive(Clone, Copy)]
    pub struct KsDecompositionParameterRanges {
        pub level: Range,
        pub log2_base: Range,
    }

    #[derive(Clone, Copy)]
    pub struct LweDimensionRange {
        pub lwe_dimension: Range,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GlweParameterRanges {
        pub log2_polynomial_size: Range,
        pub glwe_dimension: Range,
    }

    impl From<GlweParameterRanges> for LweDimensionRange {
        fn from(p: GlweParameterRanges) -> Self {
            Self {
                lwe_dimension: Range {
                    start: (1u64 << p.log2_polynomial_size.start) * p.glwe_dimension.start,
                    end: (1u64 << (p.log2_polynomial_size.end - 1)) * p.glwe_dimension.end,
                },
            }
        }
    }

    #[derive(Clone, Copy)]
    pub struct BrDecompositionParameterRanges {
        pub level: Range,
        pub log2_base: Range,
    }
}

mod grouped {
    use super::{
        BrDecompositionParameters, GlweParameters, KsDecompositionParameters, LweDimension,
    };

    #[derive(Clone)]
    pub struct Parameters {
        pub lwe_dimension: Vec<LweDimension>,
        pub glwe_dimension_and_polynomial_size: Vec<GlweParameters>,
        pub br_decomposition_parameters: Vec<BrDecompositionParameters>,
        pub ks_decomposition_parameters: Vec<KsDecompositionParameters>,
    }
}
