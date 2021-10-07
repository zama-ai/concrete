use super::{read_from_file, write_to_file};
use concrete_core::{
    crypto::secret::{GlweSecretKey, LweSecretKey},
    math::tensor::IntoTensor,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{GlweDimension, PolynomialSize};
use concrete_core::crypto::secret::generators::SecretRandomGenerator;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct RLWESecretKey {
    pub val: GlweSecretKey<BinaryKeyKind, Vec<u64>> ,
    pub polynomial_size: usize,
    pub dimension: usize,
    pub std_dev: f64,
}

impl RLWESecretKey {
    /// Generate a new secret key from an RLWEParams
    /// # Argument
    /// * `params` - an RLWEParams instance
    /// # Output
    /// * a new RLWESecretKey
    pub fn new(params: &crate::RLWEParams) -> RLWESecretKey {
        let val = GlweSecretKey::generate_binary(
            GlweDimension(params.dimension),
            PolynomialSize(params.polynomial_size),
            &mut SecretRandomGenerator::new(None),
        );
        RLWESecretKey {
            val,
            polynomial_size: params.polynomial_size,
            dimension: params.dimension,
            std_dev: params.get_std_dev(),
        }
    }

    /// Generate a new secret key from a raw dimension (i.e. without a RLWEParams input)
    /// # Argument
    /// * `polynomial_size` - the size of the polynomial
    /// * `dimension` - the length the LWE mask
    /// # Output
    /// * a new RLWESecretKey
    pub fn new_raw(polynomial_size: usize, dimension: usize, std_dev: f64) -> RLWESecretKey {
        let val = GlweSecretKey::generate_binary(
            GlweDimension(dimension),
            PolynomialSize(polynomial_size),
            &mut SecretRandomGenerator::new(None),
        );
        RLWESecretKey {
            val,
            polynomial_size,
            dimension,
            std_dev,
        }
    }

    /// Convert an RLWE secret key into an LWE secret key
    /// # Output
    /// * an LWE secret key
    pub fn to_lwe_secret_key(&self) -> crate::LWESecretKey {
        crate::LWESecretKey {
            val: LweSecretKey::binary_from_container(self.val.clone().into_tensor().into_container()),
            dimension: self.dimension * self.polynomial_size,
            std_dev: self.std_dev,
        }
    }

    /// Return the variance of the error distribution associated with the secret key
    /// # Output
    /// * the variance
    pub fn get_variance(&self) -> f64 {
        f64::powi(self.std_dev, 2i32)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<RLWESecretKey, Box<dyn Error>> {
        read_from_file(path)
    }
}

impl fmt::Display for RLWESecretKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut to_be_print: String = "".to_string();
        to_be_print = to_be_print
            + &format!(
                " RLWESecretKey {{\n         -> dimension = {}\n         -> polynomial_size = {}\n         -> std_dev = {}\n",
                self.dimension, self.polynomial_size, self.std_dev
            );
        to_be_print += "       }";

        writeln!(f, "{}", to_be_print)
    }
}
