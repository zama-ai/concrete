//! lwe_params module describing the LWEParams structure

use super::{read_from_file, write_to_file};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Structure describing the security parameters for encryption with LWE ciphertexts
/// # Attributes
/// * `dimension` -the size of an LWE mask
/// * `log2_std_dev` -the log2 of the standard deviation used for the error normal distribution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LWEParams {
    pub dimension: usize,
    pub log2_std_dev: i32,
}

//////////////////////////
// 128 bits of security //
//////////////////////////

/// 128 bits of security with a dimension of 256 (LWE estimator, September 15th 2020)
pub const LWE128_256: LWEParams = LWEParams {
    dimension: 256,
    log2_std_dev: -5,
};

/// 128 bits of security with a dimension of 512 (LWE estimator, September 15th 2020)
pub const LWE128_512: LWEParams = LWEParams {
    dimension: 512,
    log2_std_dev: -11,
};

/// 128 bits of security with a dimension of 630 (LWE estimator, September 15th 2020)
pub const LWE128_630: LWEParams = LWEParams {
    dimension: 630,
    log2_std_dev: -14,
};

/// 128 bits of security with a dimension of 650 (LWE estimator, September 15th 2020)
pub const LWE128_650: LWEParams = LWEParams {
    dimension: 650,
    log2_std_dev: -15,
};

/// 128 bits of security with a dimension of 688 (LWE estimator, September 15th 2020)
pub const LWE128_688: LWEParams = LWEParams {
    dimension: 688,
    log2_std_dev: -16,
};

/// 128 bits of security with a dimension of 710 (LWE estimator, September 15th 2020)
pub const LWE128_710: LWEParams = LWEParams {
    dimension: 710,
    log2_std_dev: -17,
};

/// 128 bits of security with a dimension of 750 (LWE estimator, September 15th 2020)
pub const LWE128_750: LWEParams = LWEParams {
    dimension: 750,
    log2_std_dev: -18,
};

/// 128 bits of security with a dimension of 800 (LWE estimator, September 15th 2020)
pub const LWE128_800: LWEParams = LWEParams {
    dimension: 800,
    log2_std_dev: -19,
};

/// 128 bits of security with a dimension of 830 (LWE estimator, September 15th 2020)
pub const LWE128_830: LWEParams = LWEParams {
    dimension: 830,
    log2_std_dev: -20,
};

/// 128 bits of security with a dimension of 1024 (LWE estimator, September 15th 2020)
pub const LWE128_1024: LWEParams = LWEParams {
    dimension: 1024,
    log2_std_dev: -25,
};

/// 128 bits of security with a dimension of 2048 (LWE estimator, September 15th 2020)
pub const LWE128_2048: LWEParams = LWEParams {
    dimension: 2048,
    log2_std_dev: -52, // warning u32
};

/// 128 bits of security with a dimension of 4096 (LWE estimator, September 15th 2020)
pub const LWE128_4096: LWEParams = LWEParams {
    dimension: 4096,
    log2_std_dev: -105, // warning u64
};

////////////////////////////////////////////////////
//                80 bits of security             //
////////////////////////////////////////////////////

/// 80 bits of security with a dimension of 256 (LWE estimator, September 15th 2020)
pub const LWE80_256: LWEParams = LWEParams {
    dimension: 256,
    log2_std_dev: -9,
};

/// 80 bits of security with a dimension of 512 (LWE estimator, September 15th 2020)
pub const LWE80_512: LWEParams = LWEParams {
    dimension: 512,
    log2_std_dev: -19,
};

/// 80 bits of security with a dimension of 630 (LWE estimator, September 15th 2020)
pub const LWE80_630: LWEParams = LWEParams {
    dimension: 630,
    log2_std_dev: -24,
};

/// 80 bits of security with a dimension of 650 (LWE estimator, September 15th 2020)
pub const LWE80_650: LWEParams = LWEParams {
    dimension: 650,
    log2_std_dev: -25,
};

/// 80 bits of security with a dimension of 688 (LWE estimator, September 15th 2020)
pub const LWE80_688: LWEParams = LWEParams {
    dimension: 688,
    log2_std_dev: -26,
};

/// 80 bits of security with a dimension of 710 (LWE estimator, September 15th 2020)
pub const LWE80_710: LWEParams = LWEParams {
    dimension: 710,
    log2_std_dev: -27,
};

/// 80 bits of security with a dimension of 750 (LWE estimator, September 15th 2020)
pub const LWE80_750: LWEParams = LWEParams {
    dimension: 750,
    log2_std_dev: -29,
};

/// 80 bits of security with a dimension of 800 (LWE estimator, September 15th 2020)
pub const LWE80_800: LWEParams = LWEParams {
    dimension: 800,
    log2_std_dev: -31, // warning u32
};

/// 80 bits of security with a dimension of 830 (LWE estimator, September 15th 2020)
pub const LWE80_830: LWEParams = LWEParams {
    dimension: 830,
    log2_std_dev: -32, // warning u32
};

/// 80 bits of security with a dimension of 1024 (LWE estimator, September 15th 2020)
pub const LWE80_1024: LWEParams = LWEParams {
    dimension: 1024,
    log2_std_dev: -40, // warning u32
};

/// 80 bits of security with a dimension of 2048 (LWE estimator, September 15th 2020)
pub const LWE80_2048: LWEParams = LWEParams {
    dimension: 2048,
    log2_std_dev: -82, // warning u64
};

impl LWEParams {
    /// Instantiate a new LWEParams with the provided dimension and standard deviation
    /// # Arguments
    /// * `dimension` -the size of an LWE mask
    /// * `std_dev` -the standard deviation used for the error normal distribution
    /// # Output
    /// * a new instantiation of an LWEParams
    pub fn new(dimension: usize, log2_std_dev: i32) -> LWEParams {
        LWEParams {
            dimension,
            log2_std_dev,
        }
    }

    pub fn get_std_dev(&self) -> f64 {
        f64::powi(2., self.log2_std_dev)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<LWEParams, Box<dyn Error>> {
        read_from_file(path)
    }
}

impl fmt::Display for LWEParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut to_be_print: String = "".to_string();
        to_be_print = to_be_print
            + &format!(
                " LWEParams {{\n         -> dimension = {}\n         -> std_dev = {}\n         -> log2_std_dev = {}\n",
                self.dimension, self.get_std_dev(), self.log2_std_dev
            );
        to_be_print += "       }";

        writeln!(f, "{}", to_be_print)
    }
}
