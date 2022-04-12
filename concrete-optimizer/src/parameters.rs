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
    pub struct PbsDecompositionParameters {
        pub level: u64,
        pub log2_base: u64,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct GlweParameters {
        pub log2_polynomial_size: u64,
        pub glwe_dimension: u64,
    }

    #[derive(Clone, Copy)]
    pub struct LweDimension {
        pub lwe_dimension: u32,
    }

    #[derive(Copy, Clone)]
    pub struct InputParameter {
        pub lwe_dimension: u32,
    }

    #[derive(Copy, Clone)]
    pub struct AtomicPatternParameters {
        pub input_lwe_dimension: LweDimension,
        pub ks_decomposition_parameter: KsDecompositionParameters,
        pub internal_lwe_dimension: LweDimension,
        pub pbs_decomposition_parameter: PbsDecompositionParameters,
        pub output_glwe_params: GlweParameters,
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
    pub struct PbsDecompositionParameterRanges {
        pub level: Range,
        pub log2_base: Range,
    }
}

mod grouped {
    use super::{
        GlweParameters, KsDecompositionParameters, LweDimension, PbsDecompositionParameters,
    };

    #[derive(Clone)]
    pub struct Parameters {
        pub lwe_dimension: Vec<LweDimension>,
        pub glwe_dimension_and_polynomial_size: Vec<GlweParameters>,
        pub pbs_decomposition_parameters: Vec<PbsDecompositionParameters>,
        pub ks_decomposition_parameters: Vec<KsDecompositionParameters>,
    }
}
