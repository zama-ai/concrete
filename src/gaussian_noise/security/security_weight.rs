#[derive(Clone, Copy)]
pub struct SecurityWeights {
    slope: f64,
    bias: f64,
    minimal_lwe_dimension: u64,
}

impl SecurityWeights {
    pub fn secure_log2_std(&self, lwe_dimension: u64, ciphertext_modulus_log: f64) -> f64 {
        // ensure to have a minimal on std deviation covering the 2 lowest bits on modular scale
        let epsilon_log2_std_modular = 2.0;
        let epsilon_log2_std = epsilon_log2_std_modular - (ciphertext_modulus_log);
        // ensure the requested lwe_dimension is bigger than the minimal lwe dimension
        if self.minimal_lwe_dimension <= lwe_dimension {
            f64::max(
                self.slope * lwe_dimension as f64 + self.bias,
                epsilon_log2_std,
            )
        } else {
            ciphertext_modulus_log
        }
    }
}

// Security curves generated using the lattice-estimator
// (https://github.com/malb/lattice-estimator) on the 24th of June 2022
const SECURITY_WEIGHTS_ARRAY: [(u64, SecurityWeights); 9] = [
    (
        80,
        SecurityWeights {
            slope: -0.040_426_331_193_645_89,
            bias: 1.660_978_864_143_672_2,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        96,
        SecurityWeights {
            slope: -0.034_147_803_608_670_51,
            bias: 2.017_310_258_660_345,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        112,
        SecurityWeights {
            slope: -0.029_670_137_081_135_885,
            bias: 2.162_463_714_083_856,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        128,
        SecurityWeights {
            slope: -0.026_405_028_765_226_22,
            bias: 2.482_642_269_104_317_7,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        144,
        SecurityWeights {
            slope: -0.023_821_437_305_989_134,
            bias: 2.717_778_944_063_667_3,
            minimal_lwe_dimension: 450,
        },
    ),
    (
        160,
        SecurityWeights {
            slope: -0.021_743_582_187_160_36,
            bias: 2.938_810_548_493_322,
            minimal_lwe_dimension: 498,
        },
    ),
    (
        176,
        SecurityWeights {
            slope: -0.019_904_056_582_117_684,
            bias: 2.816_125_280_154_224_7,
            minimal_lwe_dimension: 551,
        },
    ),
    (
        192,
        SecurityWeights {
            slope: -0.018_610_403_247_590_085,
            bias: 3.299_623_684_839_900_8,
            minimal_lwe_dimension: 606,
        },
    ),
    (
        256,
        SecurityWeights {
            slope: -0.014_606_812_351_714_953,
            bias: 3.849_362_923_469_300_3,
            minimal_lwe_dimension: 826,
        },
    ),
];

pub fn supported_security_levels() -> impl std::iter::Iterator<Item = u64> {
    SECURITY_WEIGHTS_ARRAY
        .iter()
        .map(|(security_level, _)| *security_level)
}

pub fn security_weight(security_level: u64) -> Option<SecurityWeights> {
    let index = SECURITY_WEIGHTS_ARRAY
        .binary_search_by_key(&security_level, |(security_level, _weights)| {
            *security_level
        })
        .ok()?;

    Some(SECURITY_WEIGHTS_ARRAY[index].1)
}
