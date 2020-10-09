use super::{read_from_file, write_to_file};
use crate::core_api::crypto;
use crate::core_api::math::Tensor;
use crate::crypto_api;
use crate::crypto_api::Torus;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct RLWESecretKey {
    pub val: Vec<Torus>,
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
    pub fn new(params: &crypto_api::RLWEParams) -> RLWESecretKey {
        let sk_len = <Torus as crypto::SecretKey>::get_secret_key_length(
            params.dimension,
            params.polynomial_size,
        );
        let mut sk = vec![0; sk_len];
        Tensor::uniform_random_default(&mut sk);
        RLWESecretKey {
            val: sk,
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
        let sk_len =
            <Torus as crypto::SecretKey>::get_secret_key_length(dimension, polynomial_size);
        let mut sk = vec![0; sk_len];
        Tensor::uniform_random_default(&mut sk);
        RLWESecretKey {
            val: sk,
            polynomial_size: polynomial_size,
            dimension: dimension,
            std_dev: std_dev,
        }
    }

    /// Convert an RLWE secret key into an LWE secret key
    /// # Output
    /// * an LWE secret key
    pub fn to_lwe_secret_key(&self) -> crypto_api::LWESecretKey {
        crypto_api::LWESecretKey {
            val: self.val.clone(),
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
