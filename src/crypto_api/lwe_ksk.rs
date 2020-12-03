use crate::core_api::crypto;
use crate::core_api::math::Tensor;
use crate::crypto_api;
use crate::crypto_api::Torus;
use crate::Types;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::mem::transmute;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct LWEKSK {
    pub ciphertexts: Vec<Torus>,
    pub base_log: usize,
    pub level: usize,
    pub dimension_before: usize,
    pub dimension_after: usize,
    pub variance: f64,
}

impl LWEKSK {
    /// Generate an empty LWE key switching key
    ///
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the key switch)
    /// * `sk_after` - an LWE secret key (output for the key switch)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEKSK
    pub fn zero(
        sk_before: &crypto_api::LWESecretKey,
        sk_after: &crypto_api::LWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEKSK {
        let ksk_ciphertexts: Vec<Torus> =
            vec![0; crypto::lwe::get_ksk_size(sk_before.dimension, sk_after.dimension, level)];

        LWEKSK {
            ciphertexts: ksk_ciphertexts,
            base_log: base_log,
            level: level,
            dimension_before: sk_before.dimension,
            dimension_after: sk_after.dimension,
            variance: f64::powi(sk_after.std_dev, 2),
        }
    }

    /// Generate a valid LWE key switching key
    /// # Argument
    /// * `sk_before` - an LWE secret key (input for the key switch)
    /// * `sk_after` - an LWE secret key (output for the key switch)
    /// * `base_log` - the log2 of the decomposition base
    /// * `level` - the number of levels of the decomposition
    ///
    /// # Output
    /// * an LWEKSK
    pub fn new(
        sk_before: &crypto_api::LWESecretKey,
        sk_after: &crypto_api::LWESecretKey,
        base_log: usize,
        level: usize,
    ) -> LWEKSK {
        let mut ksk_ciphertexts: Vec<Torus> =
            vec![0; crypto::lwe::get_ksk_size(sk_before.dimension, sk_after.dimension, level)];

        crypto::LWE::create_key_switching_key(
            &mut ksk_ciphertexts,
            base_log,
            level,
            sk_after.dimension,
            sk_after.std_dev,
            &sk_before.val,
            &sk_after.val,
        );

        LWEKSK {
            ciphertexts: ksk_ciphertexts,
            base_log: base_log,
            level: level,
            dimension_before: sk_before.dimension,
            dimension_after: sk_after.dimension,
            variance: f64::powi(sk_after.std_dev, 2),
        }
    }

    pub fn save(&self, path: &str) {
        let mut tensor: Vec<u64> = vec![0; self.ciphertexts.len() + 6];

        tensor[0] = self.variance.to_bits();
        tensor[1] = self.dimension_before as u64;
        tensor[2] = self.dimension_after as u64;
        tensor[3] = self.base_log as u64;
        tensor[4] = self.level as u64;
        tensor[5] = self.ciphertexts.len() as u64;

        for (dst, src) in tensor[6..(self.ciphertexts.len() + 6)]
            .iter_mut()
            .zip(self.ciphertexts.iter())
        {
            *dst = *src;
        }
        Tensor::write_in_file(&tensor, path).unwrap();
    }

    pub fn load(path: &str) -> crypto_api::LWEKSK {
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

        let mut res = crypto_api::LWEKSK {
            variance: f64::from_bits(tensor_1[0]),
            dimension_before: tensor_1[1] as usize,
            dimension_after: tensor_1[2] as usize,
            base_log: tensor_1[3] as usize,
            level: tensor_1[4] as usize,
            ciphertexts: vec![0; tensor_1[5] as usize],
        };

        for val in res.ciphertexts.iter_mut() {
            file.read(&mut bytes).unwrap();
            bytes.reverse(); // the order is wrong ...
            *val = unsafe { transmute::<[u8; <Torus as Types>::TORUS_BIT / 8], Torus>(bytes) };
        }

        res
    }
}

/// Print needed pieces of information about an LWEKSK
impl fmt::Display for LWEKSK {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " LWEBSK {\n         -> samples = [";

        if self.ciphertexts.len() <= 2 * n {
            for elt in self.ciphertexts.iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        } else {
            for elt in self.ciphertexts[0..n].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print += "...";

            for elt in self.ciphertexts[self.ciphertexts.len() - n..].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print += "]\n";
        }

        to_be_print = to_be_print + &format!("         -> variance = {}\n", self.variance);
        to_be_print =
            to_be_print + &format!("         -> dimension before = {}\n", self.dimension_before);
        to_be_print =
            to_be_print + &format!("         -> dimension after = {}\n", self.dimension_after);

        to_be_print = to_be_print + &format!("         -> base_log = {}\n", self.base_log);
        to_be_print = to_be_print + &format!("         -> level = {}\n", self.level);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
