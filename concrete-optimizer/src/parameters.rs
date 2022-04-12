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
        pub br_decomposition_parameter: BrDecompositionParameters,
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
