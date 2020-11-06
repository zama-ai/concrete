use super::{deserialize_vec_ctorus, serialize_vec_ctorus};
use super::{read_from_file, write_to_file};
use crate::core_api::crypto;
use crate::core_api::math::Tensor;
use crate::crypto_api;
use crate::crypto_api::error::CryptoAPIError;
use crate::crypto_api::Torus;
use crate::types::CTorus;
use crate::Types;
use backtrace::Backtrace;
use colored::Colorize;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::mem::transmute;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct LWEBSK {
    #[serde(
        deserialize_with = "deserialize_vec_ctorus",
        serialize_with = "serialize_vec_ctorus"
    )]
    pub ciphertexts: Vec<CTorus>,
    pub variance: f64,
    pub dimension: usize,
    pub polynomial_size: usize,
    pub base_log: usize,
    pub level: usize,
}

impl LWEBSK {
    /// Return the dimension of an LWE we can bootstrap with this key
    pub fn get_lwe_dimension(&self) -> usize {
        self.ciphertexts.len()
            / (usize::pow(self.dimension + 1, 2) * self.level * self.polynomial_size)
    }

    /// Return the log2 of the polynomial size of the RLWE involved in the bootstrap
    pub fn get_polynomial_size_log(&self) -> usize {
        f64::log2(self.polynomial_size as f64) as usize
    }

    /// Build a lookup table af a function from two encoders
    ///
    /// # Argument
    /// * `encoder_input` - the encoder of the input (of the bootstrap)
    /// * `encoder_output` - the encoder of the output (of the bootstrap)
    /// * `f` - a function
    ///
    /// # Output
    /// * a slice of Torus containing the lookup table
    pub fn generate_functional_look_up_table<F: Fn(f64) -> f64>(
        &self,
        encoder_input: &crypto_api::Encoder,
        encoder_output: &crypto_api::Encoder,
        f: F,
    ) -> Result<Vec<Torus>, CryptoAPIError> {
        // check that precision != 0
        if encoder_input.nb_bit_precision == 0 {
            return Err(PrecisionError!());
        }

        // check that the input encoder has at least 1 bit of padding
        if encoder_input.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(encoder_input.nb_bit_padding, 1));
        }

        // clone the input encoder and set nb_bit_padding to 1
        let mut encoder_input_clone = encoder_input.clone();
        encoder_input_clone.nb_bit_padding = 1;

        let mut result: Vec<Torus> = vec![0; self.polynomial_size];

        for (i, res) in result.iter_mut().enumerate() {
            // create a valid encoding from i
            let shift: usize = <Torus as Types>::TORUS_BIT - self.get_polynomial_size_log() - 1;
            let encoded: Torus = (i as Torus) << shift;

            // decode the encoding
            let decoded: f64 = encoder_input_clone.decode_operators(encoded)?;

            // apply the function
            let f_decoded: f64 = f(decoded);

            // encode the result
            let output_encoded: Torus =
                encoder_output.encode_outside_interval_operators(f_decoded)?;

            *res = output_encoded;
        }
        Ok(result)
    }

    /// Build a lookup table for the identity function from two encoders
    ///
    /// # Argument
    /// * `encoder_input` - the encoder of the input (of the bootstrap)
    /// * `encoder_output` - the encoder of the output (of the bootstrap)
    ///
    /// # Output
    /// * a slice of Torus containing the lookup table
    pub fn generate_identity_look_up_table(
        &self,
        encoder_input: &crypto_api::Encoder,
        encoder_output: &crypto_api::Encoder,
    ) -> Result<Vec<Torus>, CryptoAPIError> {
        self.generate_functional_look_up_table(encoder_input, encoder_output, |x| x)
    }

    /// Create a valid bootstrapping key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the bootstrap)
    /// * `sk_after` - an LWE secret key (output for the bootstrap)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEBSK
    pub fn new(
        sk_input: &crypto_api::LWESecretKey,
        sk_output: &crypto_api::RLWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEBSK {
        // allocation for the bootstrapping key
        let mut trgsw_ciphertexts: Vec<CTorus> = vec![
            CTorus::zero();
            crypto::cross::get_bootstrapping_key_size(
                sk_output.dimension,
                sk_output.polynomial_size,
                level,
                sk_input.dimension
            )
        ];
        crypto::RGSW::create_fourier_bootstrapping_key(
            &mut trgsw_ciphertexts,
            base_log,
            level,
            sk_output.dimension,
            sk_output.polynomial_size,
            sk_output.std_dev,
            &sk_input.val,
            &sk_output.val,
        );
        LWEBSK {
            ciphertexts: trgsw_ciphertexts,
            variance: f64::powi(sk_output.std_dev, 2),
            dimension: sk_output.dimension,
            polynomial_size: sk_output.polynomial_size,
            base_log: base_log,
            level: level,
        }
    }

    /// Create an empty bootstrapping key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the bootstrap)
    /// * `sk_after` - an LWE secret key (output for the bootstrap)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEBSK
    pub fn zero(
        sk_input: &crypto_api::LWESecretKey,
        sk_output: &crypto_api::RLWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEBSK {
        // allocation for the bootstrapping key
        let trgsw_ciphertexts: Vec<CTorus> = vec![
            CTorus::zero();
            crypto::cross::get_bootstrapping_key_size(
                sk_output.dimension,
                sk_output.polynomial_size,
                level,
                sk_input.dimension
            )
        ];
        LWEBSK {
            ciphertexts: trgsw_ciphertexts,
            variance: f64::powi(sk_output.std_dev, 2),
            dimension: sk_output.dimension,
            polynomial_size: sk_output.polynomial_size,
            base_log: base_log,
            level: level,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<LWEBSK, Box<dyn Error>> {
        read_from_file(path)
    }

    pub fn write_in_file_bytes(&self, path: &str) {
        let mut tensor: Vec<u64> = vec![0; self.ciphertexts.len() * 2 + 6];

        tensor[0] = self.variance.to_bits();
        tensor[1] = self.dimension as u64;
        tensor[2] = self.polynomial_size as u64;
        tensor[3] = self.base_log as u64;
        tensor[4] = self.level as u64;
        tensor[5] = self.ciphertexts.len() as u64;

        for (couple, c) in tensor[6..(self.ciphertexts.len() * 2 + 6)]
            .chunks_mut(2)
            .zip(self.ciphertexts.iter())
        {
            couple[0] = c.re.to_bits();
            couple[1] = c.im.to_bits();
        }
        Tensor::write_in_file(&tensor, path).unwrap();
    }

    pub fn read_in_file_bytes(path: &str) -> crypto_api::LWEBSK {
        let mut tensor_1: Vec<u64> = vec![0; 6];
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .unwrap();

        let mut bytes: [u8; <Torus as Types>::TORUS_BIT / 8] = [0; <Torus as Types>::TORUS_BIT / 8];

        for val in tensor_1.iter_mut() {
            file.read(&mut bytes).unwrap();
            bytes.reverse(); // the order is wrong ...
            *val = unsafe { transmute::<[u8; <Torus as Types>::TORUS_BIT / 8], Torus>(bytes) };
        }

        let mut res = crypto_api::LWEBSK {
            variance: f64::from_bits(tensor_1[0]),
            dimension: tensor_1[1] as usize,
            polynomial_size: tensor_1[2] as usize,
            base_log: tensor_1[3] as usize,
            level: tensor_1[4] as usize,
            ciphertexts: vec![CTorus::zero(); tensor_1[5] as usize],
        };

        let mut tensor_2: Vec<u64> = vec![0; (tensor_1[5] * 2) as usize];

        for val in tensor_2.iter_mut() {
            file.read(&mut bytes).unwrap();
            bytes.reverse(); // the order is wrong ...
            *val = unsafe { transmute::<[u8; <Torus as Types>::TORUS_BIT / 8], Torus>(bytes) };
        }

        for (couple, c) in tensor_2.chunks(2).zip(res.ciphertexts.iter_mut()) {
            *c = CTorus::new(f64::from_bits(couple[0]), f64::from_bits(couple[1]));
        }
        res
    }
}

/// Print needed pieces of information about an LWEBSK
impl fmt::Display for LWEBSK {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " LWEBSK {\n         -> samples = [";

        if self.ciphertexts.len() <= 2 * n {
            for elt in self.ciphertexts.iter() {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        } else {
            for elt in self.ciphertexts[0..n].iter() {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "...";

            for elt in self.ciphertexts[self.ciphertexts.len() - n..].iter() {
                to_be_print += &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        }

        to_be_print += &format!("         -> variance = {}\n", self.variance);
        to_be_print += &format!("         -> dimension = {}\n", self.dimension);
        to_be_print =
            to_be_print + &format!("         -> polynomial_size = {}\n", self.polynomial_size);
        to_be_print += &format!("         -> base_log = {}\n", self.base_log);
        to_be_print += &format!("         -> level = {}\n", self.level);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
