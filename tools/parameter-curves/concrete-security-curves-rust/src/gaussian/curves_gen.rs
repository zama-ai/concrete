use super::security_weights::SecurityWeights;
pub const SECURITY_WEIGHTS_ARRAY: [(u64, SecurityWeights); 2] = [
    (
        128,
        SecurityWeights {
            slope: -0.025696778711484593,
            bias: 2.675931372549016,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        132,
        SecurityWeights {
            slope: -0.024891456582633045,
            bias: 2.65734593837534,
            minimal_lwe_dimension: 450,
        },
    ),
];
