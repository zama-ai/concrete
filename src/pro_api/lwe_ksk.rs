use super::{read_from_file, write_to_file};
use crate::operators::crypto;
use crate::pro_api;
use crate::pro_api::Torus;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

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
        sk_before: &pro_api::LWESecretKey,
        sk_after: &pro_api::LWESecretKey,
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
        sk_before: &pro_api::LWESecretKey,
        sk_after: &pro_api::LWESecretKey,
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
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<LWEKSK, Box<dyn Error>> {
        read_from_file(path)
    }
}

/// Print needed pieces of information about an LWEKSK
impl fmt::Display for LWEKSK {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print = to_be_print + " LWEBSK {\n         -> samples = [";

        if self.ciphertexts.len() <= 2 * n {
            for elt in self.ciphertexts.iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print = to_be_print + "]\n";
        } else {
            for elt in self.ciphertexts[0..n].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print = to_be_print + "...";

            for elt in self.ciphertexts[self.ciphertexts.len() - n..].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print = to_be_print + "]\n";
        }

        to_be_print = to_be_print + &format!("         -> variance = {}\n", self.variance);
        to_be_print =
            to_be_print + &format!("         -> dimension before = {}\n", self.dimension_before);
        to_be_print =
            to_be_print + &format!("         -> dimension after = {}\n", self.dimension_after);

        to_be_print = to_be_print + &format!("         -> base_log = {}\n", self.base_log);
        to_be_print = to_be_print + &format!("         -> level = {}\n", self.level);
        to_be_print = to_be_print + "       }";
        writeln!(f, "{}", to_be_print)
    }
}
