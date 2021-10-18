use super::{read_from_file, write_to_file};
use crate::error::CryptoAPIError;
use backtrace::Backtrace;
use colored::Colorize;
use concrete_core::{
    crypto::secret::{GlweSecretKey, LweSecretKey},
    math::tensor::IntoTensor,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{LweDimension, PolynomialSize};
use concrete_core::crypto::secret::generators::SecretRandomGenerator;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LWESecretKey {
    pub val: LweSecretKey<BinaryKeyKind, Vec<u64>>,
    pub dimension: usize,
    pub std_dev: f64,
}

impl LWESecretKey {
    /// Generate a new secret key from an LWEParams
    /// # Argument
    /// * `p` - an LWEParams instance
    /// # Output
    /// * a new LWESecretKey
    pub fn new(params: &crate::LWEParams) -> LWESecretKey {
        let val =  LweSecretKey::generate_binary(LweDimension(params.dimension), &mut SecretRandomGenerator::new
            (None));
        LWESecretKey {
            val,
            dimension: params.dimension,
            std_dev: params.get_std_dev(),
        }
    }

    /// Generate a new secret key from a raw dimension (i.e. without a LWEParams input)
    /// # Argument
    /// * `dimension` s the length the LWE mask
    /// * `std_dev` - the standard deviation for the encryption
    /// # Output
    /// * a new LWESecretKey
    pub fn new_raw(dimension: usize, std_dev: f64) -> LWESecretKey {
        let val = LweSecretKey::generate_binary(LweDimension(dimension), &mut SecretRandomGenerator::new
            (None));
        LWESecretKey {
            val,
            dimension,
            std_dev,
        }
    }

    /// Convert an LWE secret key into an RLWE secret key
    /// # Input
    /// * `polynomial_size` - the size of the polynomial of the output RLWE secret key
    /// # Output
    /// * an RLWE secret key
    pub fn to_rlwe_secret_key(
        &self,
        polynomial_size: usize,
    ) -> Result<crate::RLWESecretKey, CryptoAPIError> {
        if self.dimension % polynomial_size != 0 {
            return Err(LweToRlweError!(self.dimension, polynomial_size));
        }
        Ok(crate::RLWESecretKey {
            val: GlweSecretKey::binary_from_container(
                self.val.clone().into_tensor().into_container(),
                PolynomialSize(polynomial_size),
            ),
            dimension: self.dimension / polynomial_size,
            polynomial_size,
            std_dev: self.std_dev,
        })
    }

    /// Return the variance of the error distribution associated with the secret key
    /// # Output
    /// * a variance
    pub fn get_variance(&self) -> f64 {
        f64::powi(self.std_dev, 2i32)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<LWESecretKey, Box<dyn Error>> {
        read_from_file(path)
    }
}

impl fmt::Display for LWESecretKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut to_be_print: String = "".to_string();
        to_be_print = to_be_print
            + &format!(
                " LWESecretKey {{\n         -> dimension = {}\n         -> std_dev = {}\n",
                self.dimension, self.std_dev
            );
        to_be_print += "       }";

        writeln!(f, "{}", to_be_print)
    }
}
