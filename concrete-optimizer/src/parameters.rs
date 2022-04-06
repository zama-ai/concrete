pub use grouped::*;
pub use individual::*;
pub use range::*;

mod individual {

    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct KsDecompositionParameters {
        pub level: u64,
        pub log2_base: u64,
    }

    #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
    pub struct BrDecompositionParameters {
        pub level: u64,
        pub log2_base: u64,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct GlweParameters {
        pub log2_polynomial_size: u64,
        pub glwe_dimension: u64,
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
