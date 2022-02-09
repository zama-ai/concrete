pub use grouped::*;
pub use individual::*;

mod individual {
    #[derive(Clone, Copy)]
    pub struct KsDecompositionParameters<LogBase, Level> {
        pub log2_base: LogBase,
        pub level: Level,
    }

    #[derive(Clone, Copy)]
    pub struct PbsDecompositionParameters<LogBase, Level> {
        pub log2_base: LogBase,
        pub level: Level,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub struct GlweParameters<LogPolynomialSize, GlweDimension> {
        pub log2_polynomial_size: LogPolynomialSize,
        pub glwe_dimension: GlweDimension,
    }

    #[derive(Clone, Copy)]
    pub struct LweDimension<LweDimension2> {
        pub lwe_dimension: LweDimension2,
    }

    #[derive(Copy, Clone)]
    pub struct InputParameter<LweDimension> {
        pub lwe_dimension: LweDimension,
    }

    #[derive(Copy, Clone)]
    pub struct AtomicPatternParameters<
        InputLweDimension,
        KsDecompositionParameter,
        InternalLweDimension,
        PbsDecompositionParameter,
        GlweDimensionAndPolynomialSize,
    > {
        pub input_lwe_dimension: InputLweDimension,
        pub ks_decomposition_parameter: KsDecompositionParameter,
        pub internal_lwe_dimension: InternalLweDimension,
        pub pbs_decomposition_parameter: PbsDecompositionParameter,
        pub output_glwe_params: GlweDimensionAndPolynomialSize,
    }
}

mod grouped {
    use super::{
        GlweParameters, KsDecompositionParameters, LweDimension, PbsDecompositionParameters,
    };

    #[derive(Clone)]
    pub struct Parameters<
        LweDimension2,
        KsLogBase,
        KsLevel,
        PbsLogBase,
        PbsLevel,
        LogPolynomialSize,
        GlweDimension,
    > {
        pub lwe_dimension: Vec<LweDimension<LweDimension2>>,
        pub glwe_dimension_and_polynomial_size:
            Vec<GlweParameters<LogPolynomialSize, GlweDimension>>,
        pub pbs_decomposition_parameters: Vec<PbsDecompositionParameters<PbsLogBase, PbsLevel>>,
        pub ks_decomposition_parameters: Vec<KsDecompositionParameters<KsLogBase, KsLevel>>,
    }
}
