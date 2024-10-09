use crate::parameters::{
    BrDecompositionParameterRanges, GlweParameterRanges, KsDecompositionParameterRanges,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
struct ParameterCount {
    pub glwe: usize,
    pub br_decomposition: usize,
    pub ks_decomposition: usize,
}

#[derive(Clone, Copy)]
pub struct ParameterDomains {
    // move next comment to pareto ranges definition
    // TODO: verify if pareto optimal parameters depends on precisions
    pub glwe_pbs_constrained_cpu: GlweParameterRanges,
    pub glwe_pbs_constrained_gpu: GlweParameterRanges,
    pub free_glwe: GlweParameterRanges,
    pub br_decomposition: BrDecompositionParameterRanges,
    pub ks_decomposition: KsDecompositionParameterRanges,
    pub free_lwe: Range,
}

pub const DEFAULT_DOMAINS: ParameterDomains = ParameterDomains {
    glwe_pbs_constrained_cpu: GlweParameterRanges {
        log2_polynomial_size: Range { start: 8, end: 18 },
        glwe_dimension: Range { start: 1, end: 7 },
    },
    glwe_pbs_constrained_gpu: GlweParameterRanges {
        log2_polynomial_size: Range { start: 8, end: 14 },
        glwe_dimension: Range { start: 1, end: 7 },
    },
    free_glwe: GlweParameterRanges {
        log2_polynomial_size: Range { start: 0, end: 1 },
        glwe_dimension: Range {
            start: 512,
            end: 2048,
        },
    },
    br_decomposition: BrDecompositionParameterRanges {
        log2_base: Range { start: 1, end: 65 },
        level: Range { start: 1, end: 65 },
    },
    ks_decomposition: KsDecompositionParameterRanges {
        log2_base: Range { start: 1, end: 65 },
        level: Range { start: 1, end: 65 },
    },
    free_lwe: Range {
        start: 512,
        end: 1 << 20,
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub start: u64,
    pub end: u64,
}

#[allow(unknown_lints)]
#[allow(clippy::into_iter_without_iter)]
impl IntoIterator for &Range {
    type Item = u64;

    type IntoIter = std::ops::Range<u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.start..self.end
    }
}

impl Range {
    pub fn as_vec(self) -> Vec<u64> {
        self.into_iter().collect()
    }
}
