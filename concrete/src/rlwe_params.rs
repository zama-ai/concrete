//! rlwe_params module describing the RLWEParams structure
use super::{read_from_file, write_to_file};
use crate::error::CryptoAPIError;
use backtrace::Backtrace;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Structure describing the security parameters for encryption with RLWE ciphertexts
/// # Attributes
/// - `polynomial_size`: the number of coefficients in a polynomial
/// - `dimension`: the size of an RLWE mask
/// - `log2_std_dev`: the log2 of the standard deviation used for the error normal distribution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RLWEParams {
    pub polynomial_size: usize,
    pub dimension: usize,
    pub log2_std_dev: i32,
}

////////////////////////////////////////
// 128 bits of security - dimension 1 //
////////////////////////////////////////

/// 128 bits of security with a polynomial_size of 1 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE128_256_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 256,
    log2_std_dev: -5,
};
/// 128 bits of security with a polynomial_size of 1 and a polynomial size of 512 (LWE estimator, September 15th 2020)
pub const RLWE128_512_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 512,
    log2_std_dev: -11,
};
/// 128 bits of security with a polynomial_size of 1 and a polynomial size of 1024 (LWE estimator, September 15th 2020)
pub const RLWE128_1024_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 1024,
    log2_std_dev: -25,
};
/// 128 bits of security with a polynomial_size of 1 and a polynomial size of 2048 (LWE estimator, September 15th 2020)
pub const RLWE128_2048_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 2048,
    log2_std_dev: -52, // warning u32
};
/// 128 bits of security with a polynomial_size of 1 and a polynomial size of 4096 (LWE estimator, September 15th 2020)
pub const RLWE128_4096_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 4096,
    log2_std_dev: -105, // warning u64
};

////////////////////////////////////////
// 128 bits of security - dimension 2 //
////////////////////////////////////////

/// 128 bits of security with a polynomial_size of 2 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE128_256_2: RLWEParams = RLWEParams {
    dimension: 2,
    polynomial_size: 256,
    log2_std_dev: -11,
};
/// 128 bits of security with a polynomial_size of 2 and a polynomial size of 512 (LWE estimator, September 15th 2020)
pub const RLWE128_512_2: RLWEParams = RLWEParams {
    dimension: 2,
    polynomial_size: 512,
    log2_std_dev: -25,
};

////////////////////////////////////////
// 128 bits of security - dimension 4 //
////////////////////////////////////////

/// 128 bits of security with a polynomial_size of 4 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE128_256_4: RLWEParams = RLWEParams {
    dimension: 4,
    polynomial_size: 256,
    log2_std_dev: -25,
};

///////////////////////////////////////
// 80 bits of security - dimension 1 //
///////////////////////////////////////

/// 80 bits of security with a polynomial_size of 1 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE80_256_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 256,
    log2_std_dev: -9,
};
/// 80 bits of security with a polynomial_size of 1 and a polynomial size of 512 (LWE estimator, September 15th 2020)
pub const RLWE80_512_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 512,
    log2_std_dev: -19,
};
/// 80 bits of security with a polynomial_size of 1 and a polynomial size of 1024 (LWE estimator, September 15th 2020)
pub const RLWE80_1024_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 1024,
    log2_std_dev: -40, // warning u32
};
/// 80 bits of security with a polynomial_size of 1 and a polynomial size of 2048 (LWE estimator, September 15th 2020)
pub const RLWE80_2048_1: RLWEParams = RLWEParams {
    dimension: 1,
    polynomial_size: 2048,
    log2_std_dev: -82, // warning u64
};

///////////////////////////////////////
// 80 bits of security - dimension 2 //
///////////////////////////////////////

/// 80 bits of security with a polynomial_size of 2 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE80_256_2: RLWEParams = RLWEParams {
    dimension: 2,
    polynomial_size: 256,
    log2_std_dev: -19,
};
/// 80 bits of security with a polynomial_size of 2 and a polynomial size of 512 (LWE estimator, September 15th 2020)
pub const RLWE80_512_2: RLWEParams = RLWEParams {
    dimension: 2,
    polynomial_size: 512,
    log2_std_dev: -40, // warning u32
};

///////////////////////////////////////
// 80 bits of security - dimension 4 //
///////////////////////////////////////

/// 80 bits of security with a polynomial_size of 4 and a polynomial size of 256 (LWE estimator, September 15th 2020)
pub const RLWE80_256_4: RLWEParams = RLWEParams {
    dimension: 4,
    polynomial_size: 256,
    log2_std_dev: -40, // warning u32
};

impl RLWEParams {
    /// Instantiate a new RLWEParams with the provided dimension and standard deviation
    /// # Arguments
    /// * `polynomial_size` - the number of coefficients in a polynomial
    /// * `dimension` - the size of an RLWE mask
    /// * `std_dev` - the standard deviation used for the error normal distribution
    /// # Output
    /// * a new instantiation of an RLWEParams
    /// * NotPowerOfTwoError if `polynomial_size` is not a power of 2

    pub fn new(
        polynomial_size: usize,
        dimension: usize,
        log2_std_dev: i32,
    ) -> Result<RLWEParams, CryptoAPIError> {
        if (polynomial_size as f64 - f64::powi(2., (polynomial_size as f64).log2().round() as i32))
            .abs()
            > f64::EPSILON
        {
            return Err(NotPowerOfTwoError!(polynomial_size));
        }
        Ok(RLWEParams {
            polynomial_size,
            dimension,
            log2_std_dev,
        })
    }

    pub fn get_std_dev(&self) -> f64 {
        f64::powi(2., self.log2_std_dev)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<RLWEParams, Box<dyn Error>> {
        read_from_file(path)
    }
}

impl fmt::Display for RLWEParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut to_be_print: String = "".to_string();
        to_be_print = to_be_print
            + &format!(
                " RLWEParams {{\n         -> dimension = {}\n         -> std_dev = {}\n         -> log2_std_dev = {}\n         -> polynomial_size = {}\n",
                self.dimension, self.get_std_dev(),self.log2_std_dev, self.polynomial_size
            );
        to_be_print += "       }";

        writeln!(f, "{}", to_be_print)
    }
}
