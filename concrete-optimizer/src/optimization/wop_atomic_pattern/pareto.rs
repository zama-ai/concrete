use crate::parameters::{BrDecompositionParameters, KsDecompositionParameters};

#[rustfmt::skip]
pub const BR_PARETO_DECOMP: &[BrDecompositionParameters; 71] = &[BrDecompositionParameters { level: 1, log2_base: 2 }, BrDecompositionParameters { level: 1, log2_base: 5 }, BrDecompositionParameters { level: 1, log2_base: 9 }, BrDecompositionParameters { level: 1, log2_base: 12 }, BrDecompositionParameters { level: 1, log2_base: 19 }, BrDecompositionParameters { level: 1, log2_base: 22 }, BrDecompositionParameters { level: 1, log2_base: 23 }, BrDecompositionParameters { level: 1, log2_base: 24 }, BrDecompositionParameters { level: 2, log2_base: 1 }, BrDecompositionParameters { level: 2, log2_base: 3 }, BrDecompositionParameters { level: 2, log2_base: 6 }, BrDecompositionParameters { level: 2, log2_base: 8 }, BrDecompositionParameters { level: 2, log2_base: 12 }, BrDecompositionParameters { level: 2, log2_base: 15 }, BrDecompositionParameters { level: 2, log2_base: 16 }, BrDecompositionParameters { level: 3, log2_base: 3 }, BrDecompositionParameters { level: 3, log2_base: 4 }, BrDecompositionParameters { level: 3, log2_base: 6 }, BrDecompositionParameters { level: 3, log2_base: 9 }, BrDecompositionParameters { level: 3, log2_base: 11 }, BrDecompositionParameters { level: 3, log2_base: 12 }, BrDecompositionParameters { level: 4, log2_base: 2 }, BrDecompositionParameters { level: 4, log2_base: 4 }, BrDecompositionParameters { level: 4, log2_base: 5 }, BrDecompositionParameters { level: 4, log2_base: 8 }, BrDecompositionParameters { level: 4, log2_base: 9 }, BrDecompositionParameters { level: 4, log2_base: 10 }, BrDecompositionParameters { level: 5, log2_base: 2 }, BrDecompositionParameters { level: 5, log2_base: 3 }, BrDecompositionParameters { level: 5, log2_base: 4 }, BrDecompositionParameters { level: 5, log2_base: 6 }, BrDecompositionParameters { level: 5, log2_base: 8 }, BrDecompositionParameters { level: 6, log2_base: 6 }, BrDecompositionParameters { level: 6, log2_base: 7 }, BrDecompositionParameters { level: 7, log2_base: 2 }, BrDecompositionParameters { level: 7, log2_base: 3 }, BrDecompositionParameters { level: 7, log2_base: 5 }, BrDecompositionParameters { level: 7, log2_base: 6 }, BrDecompositionParameters { level: 8, log2_base: 1 }, BrDecompositionParameters { level: 8, log2_base: 2 }, BrDecompositionParameters { level: 8, log2_base: 4 }, BrDecompositionParameters { level: 8, log2_base: 5 }, BrDecompositionParameters { level: 9, log2_base: 1 }, BrDecompositionParameters { level: 9, log2_base: 4 }, BrDecompositionParameters { level: 9, log2_base: 5 }, BrDecompositionParameters { level: 10, log2_base: 2 }, BrDecompositionParameters { level: 10, log2_base: 4 }, BrDecompositionParameters { level: 11, log2_base: 2 }, BrDecompositionParameters { level: 11, log2_base: 3 }, BrDecompositionParameters { level: 11, log2_base: 4 }, BrDecompositionParameters { level: 12, log2_base: 3 }, BrDecompositionParameters { level: 14, log2_base: 3 }, BrDecompositionParameters { level: 15, log2_base: 1 }, BrDecompositionParameters { level: 15, log2_base: 3 }, BrDecompositionParameters { level: 16, log2_base: 1 }, BrDecompositionParameters { level: 17, log2_base: 2 }, BrDecompositionParameters { level: 18, log2_base: 2 }, BrDecompositionParameters { level: 21, log2_base: 1 }, BrDecompositionParameters { level: 21, log2_base: 2 }, BrDecompositionParameters { level: 22, log2_base: 1 }, BrDecompositionParameters { level: 22, log2_base: 2 }, BrDecompositionParameters { level: 23, log2_base: 1 }, BrDecompositionParameters { level: 23, log2_base: 2 }, BrDecompositionParameters { level: 34, log2_base: 1 }, BrDecompositionParameters { level: 35, log2_base: 1 }, BrDecompositionParameters { level: 36, log2_base: 1 }, BrDecompositionParameters { level: 42, log2_base: 1 }, BrDecompositionParameters { level: 43, log2_base: 1 }, BrDecompositionParameters { level: 44, log2_base: 1 }, BrDecompositionParameters { level: 45, log2_base: 1 }, BrDecompositionParameters { level: 46, log2_base: 1 }];

#[rustfmt::skip]
pub const KS_PARETO_DECOMP: &[KsDecompositionParameters; 50] = &[KsDecompositionParameters { level: 1, log2_base: 5 }, KsDecompositionParameters { level: 1, log2_base: 6 }, KsDecompositionParameters { level: 1, log2_base: 7 }, KsDecompositionParameters { level: 1, log2_base: 8 }, KsDecompositionParameters { level: 1, log2_base: 9 }, KsDecompositionParameters { level: 1, log2_base: 10 }, KsDecompositionParameters { level: 1, log2_base: 11 }, KsDecompositionParameters { level: 1, log2_base: 12 }, KsDecompositionParameters { level: 2, log2_base: 4 }, KsDecompositionParameters { level: 2, log2_base: 5 }, KsDecompositionParameters { level: 2, log2_base: 6 }, KsDecompositionParameters { level: 2, log2_base: 7 }, KsDecompositionParameters { level: 2, log2_base: 8 }, KsDecompositionParameters { level: 3, log2_base: 3 }, KsDecompositionParameters { level: 3, log2_base: 4 }, KsDecompositionParameters { level: 3, log2_base: 5 }, KsDecompositionParameters { level: 3, log2_base: 6 }, KsDecompositionParameters { level: 4, log2_base: 2 }, KsDecompositionParameters { level: 4, log2_base: 3 }, KsDecompositionParameters { level: 4, log2_base: 4 }, KsDecompositionParameters { level: 4, log2_base: 5 }, KsDecompositionParameters { level: 5, log2_base: 2 }, KsDecompositionParameters { level: 5, log2_base: 3 }, KsDecompositionParameters { level: 5, log2_base: 4 }, KsDecompositionParameters { level: 6, log2_base: 2 }, KsDecompositionParameters { level: 6, log2_base: 3 }, KsDecompositionParameters { level: 6, log2_base: 4 }, KsDecompositionParameters { level: 7, log2_base: 2 }, KsDecompositionParameters { level: 7, log2_base: 3 }, KsDecompositionParameters { level: 8, log2_base: 2 }, KsDecompositionParameters { level: 8, log2_base: 3 }, KsDecompositionParameters { level: 9, log2_base: 1 }, KsDecompositionParameters { level: 9, log2_base: 2 }, KsDecompositionParameters { level: 10, log2_base: 1 }, KsDecompositionParameters { level: 10, log2_base: 2 }, KsDecompositionParameters { level: 11, log2_base: 1 }, KsDecompositionParameters { level: 11, log2_base: 2 }, KsDecompositionParameters { level: 12, log2_base: 1 }, KsDecompositionParameters { level: 12, log2_base: 2 }, KsDecompositionParameters { level: 13, log2_base: 1 }, KsDecompositionParameters { level: 14, log2_base: 1 }, KsDecompositionParameters { level: 15, log2_base: 1 }, KsDecompositionParameters { level: 16, log2_base: 1 }, KsDecompositionParameters { level: 17, log2_base: 1 }, KsDecompositionParameters { level: 18, log2_base: 1 }, KsDecompositionParameters { level: 19, log2_base: 1 }, KsDecompositionParameters { level: 20, log2_base: 1 }, KsDecompositionParameters { level: 21, log2_base: 1 }, KsDecompositionParameters { level: 22, log2_base: 1 }, KsDecompositionParameters { level: 23, log2_base: 1 }];
#[rustfmt::skip]
pub const KS_CIRCUIT_BOOTSTRAP_PARETO_DECOMP: &[KsDecompositionParameters; 27] = &[
    KsDecompositionParameters {
        level: 1,
        log2_base: 4,
    },
    KsDecompositionParameters {
        level: 1,
        log2_base: 5,
    },
    KsDecompositionParameters {
        level: 1,
        log2_base: 6,
    },
    KsDecompositionParameters {
        level: 1,
        log2_base: 7,
    },
    KsDecompositionParameters {
        level: 1,
        log2_base: 8,
    },
    KsDecompositionParameters {
        level: 2,
        log2_base: 3,
    },
    KsDecompositionParameters {
        level: 2,
        log2_base: 4,
    },
    KsDecompositionParameters {
        level: 2,
        log2_base: 5,
    },
    KsDecompositionParameters {
        level: 3,
        log2_base: 2,
    },
    KsDecompositionParameters {
        level: 3,
        log2_base: 3,
    },
    KsDecompositionParameters {
        level: 3,
        log2_base: 4,
    },
    KsDecompositionParameters {
        level: 4,
        log2_base: 2,
    },
    KsDecompositionParameters {
        level: 4,
        log2_base: 3,
    },
    KsDecompositionParameters {
        level: 5,
        log2_base: 2,
    },
    KsDecompositionParameters {
        level: 5,
        log2_base: 3,
    },
    KsDecompositionParameters {
        level: 6,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 6,
        log2_base: 2,
    },
    KsDecompositionParameters {
        level: 7,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 7,
        log2_base: 2,
    },
    KsDecompositionParameters {
        level: 8,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 9,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 10,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 11,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 12,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 13,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 14,
        log2_base: 1,
    },
    KsDecompositionParameters {
        level: 15,
        log2_base: 1,
    },
];

#[rustfmt::skip]
pub const BR_CIRCUIT_BOOTSTRAP_PARETO_DECOMP: &[BrDecompositionParameters; 75] = &[BrDecompositionParameters { level: 1, log2_base: 1 }, BrDecompositionParameters { level: 1, log2_base: 2 }, BrDecompositionParameters { level: 1, log2_base: 3 }, BrDecompositionParameters { level: 1, log2_base: 4 }, BrDecompositionParameters { level: 1, log2_base: 5 }, BrDecompositionParameters { level: 1, log2_base: 6 }, BrDecompositionParameters { level: 2, log2_base: 1 }, BrDecompositionParameters { level: 2, log2_base: 2 }, BrDecompositionParameters { level: 2, log2_base: 3 }, BrDecompositionParameters { level: 2, log2_base: 4 }, BrDecompositionParameters { level: 2, log2_base: 5 }, BrDecompositionParameters { level: 3, log2_base: 1 }, BrDecompositionParameters { level: 3, log2_base: 2 }, BrDecompositionParameters { level: 3, log2_base: 3 }, BrDecompositionParameters { level: 3, log2_base: 4 }, BrDecompositionParameters { level: 3, log2_base: 5 }, BrDecompositionParameters { level: 3, log2_base: 6 }, BrDecompositionParameters { level: 4, log2_base: 1 }, BrDecompositionParameters { level: 4, log2_base: 2 }, BrDecompositionParameters { level: 4, log2_base: 3 }, BrDecompositionParameters { level: 4, log2_base: 4 }, BrDecompositionParameters { level: 4, log2_base: 5 }, BrDecompositionParameters { level: 5, log2_base: 1 }, BrDecompositionParameters { level: 5, log2_base: 2 }, BrDecompositionParameters { level: 5, log2_base: 3 }, BrDecompositionParameters { level: 5, log2_base: 4 }, BrDecompositionParameters { level: 6, log2_base: 1 }, BrDecompositionParameters { level: 6, log2_base: 2 }, BrDecompositionParameters { level: 6, log2_base: 3 }, BrDecompositionParameters { level: 6, log2_base: 4 }, BrDecompositionParameters { level: 7, log2_base: 1 }, BrDecompositionParameters { level: 7, log2_base: 2 }, BrDecompositionParameters { level: 7, log2_base: 3 }, BrDecompositionParameters { level: 7, log2_base: 4 }, BrDecompositionParameters { level: 8, log2_base: 1 }, BrDecompositionParameters { level: 8, log2_base: 2 }, BrDecompositionParameters { level: 8, log2_base: 3 }, BrDecompositionParameters { level: 9, log2_base: 1 }, BrDecompositionParameters { level: 9, log2_base: 2 }, BrDecompositionParameters { level: 9, log2_base: 3 }, BrDecompositionParameters { level: 10, log2_base: 1 }, BrDecompositionParameters { level: 10, log2_base: 2 }, BrDecompositionParameters { level: 10, log2_base: 3 }, BrDecompositionParameters { level: 11, log2_base: 1 }, BrDecompositionParameters { level: 11, log2_base: 2 }, BrDecompositionParameters { level: 12, log2_base: 1 }, BrDecompositionParameters { level: 12, log2_base: 2 }, BrDecompositionParameters { level: 13, log2_base: 1 }, BrDecompositionParameters { level: 13, log2_base: 2 }, BrDecompositionParameters { level: 14, log2_base: 1 }, BrDecompositionParameters { level: 14, log2_base: 2 }, BrDecompositionParameters { level: 15, log2_base: 1 }, BrDecompositionParameters { level: 15, log2_base: 2 }, BrDecompositionParameters { level: 16, log2_base: 1 }, BrDecompositionParameters { level: 16, log2_base: 2 }, BrDecompositionParameters { level: 17, log2_base: 1 }, BrDecompositionParameters { level: 17, log2_base: 2 }, BrDecompositionParameters { level: 18, log2_base: 1 }, BrDecompositionParameters { level: 19, log2_base: 1 }, BrDecompositionParameters { level: 20, log2_base: 1 }, BrDecompositionParameters { level: 21, log2_base: 1 }, BrDecompositionParameters { level: 22, log2_base: 1 }, BrDecompositionParameters { level: 23, log2_base: 1 }, BrDecompositionParameters { level: 24, log2_base: 1 }, BrDecompositionParameters { level: 25, log2_base: 1 }, BrDecompositionParameters { level: 26, log2_base: 1 }, BrDecompositionParameters { level: 27, log2_base: 1 }, BrDecompositionParameters { level: 28, log2_base: 1 }, BrDecompositionParameters { level: 29, log2_base: 1 }, BrDecompositionParameters { level: 30, log2_base: 1 }, BrDecompositionParameters { level: 31, log2_base: 1 }, BrDecompositionParameters { level: 32, log2_base: 1 }, BrDecompositionParameters { level: 33, log2_base: 1 }, BrDecompositionParameters { level: 34, log2_base: 1 }, BrDecompositionParameters { level: 35, log2_base: 1 }];
