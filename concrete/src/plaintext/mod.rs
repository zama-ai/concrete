//! plaintext module
use super::{read_from_file, write_to_file};
use crate::error::CryptoAPIError;
use crate::Torus;
use backtrace::Backtrace;
use colored::Colorize;
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Structure describing a list of plaintext values with their respective Encoder
/// # Attributes
/// * `encoder` - the list of the encoders (one for each plaintext)
/// * `plaintexts` - the list of plaintexts
/// * `nb_plaintexts` - the size of both lists
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Plaintext {
    pub encoders: Vec<crate::Encoder>,
    pub plaintexts: Vec<Torus>,
    pub nb_plaintexts: usize,
}

impl Plaintext {
    /// Instantiate a new empty Plaintext (set to zero) of a certain size
    /// # Argument
    /// * `nb_plaintexts` - the number of plaintext that would be in the Plaintext instance
    /// # Output
    /// * a new instantiation of an empty Plaintext (set to zero) of a certain size
    /// # Example
    /// ```rust
    /// use concrete::Plaintext;
    /// let nb_ct: usize = 100;
    /// let plaintexts = Plaintext::zero(nb_ct);
    /// ```
    pub fn zero(nb_plaintexts: usize) -> Plaintext {
        Plaintext {
            encoders: vec![crate::Encoder::zero(); nb_plaintexts],
            plaintexts: vec![0; nb_plaintexts],
            nb_plaintexts,
        }
    }

    /// Instantiate a new Plaintext filled with plaintexts
    /// # Argument
    /// * `messages`- a list of messages as u64
    /// * `encoder`- an encoder
    /// # Output
    /// * a new instance of Plaintext containing the plaintext of each message with respect to encoder
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list of messages in our interval
    /// let messages: Vec<f64> = vec![-3.2, 4.3, 0.12, -1.1, 2.78];
    ///
    /// // create a new Plaintext instance filled with the plaintexts we want
    /// let pt = Plaintext::encode(&messages, &encoder).unwrap();
    ///
    /// // print the Plaintext
    /// println!("ec = {}", pt);
    /// ```
    pub fn encode(messages: &[f64], encoder: &crate::Encoder) -> Result<Plaintext, CryptoAPIError> {
        let mut res = Plaintext {
            encoders: vec![encoder.clone(); messages.len()],
            plaintexts: vec![0; messages.len()],
            nb_plaintexts: messages.len(),
        };
        res.encode_inplace(messages)?;
        Ok(res)
    }

    /// Decode one single plaintext (from the list of plaintexts in this Plaintext instance) according to its own encoder
    /// # Arguments
    /// * `nth` - the index of the plaintext to decode
    /// # Output
    /// * the decoded value as a f64
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list of messages in our interval
    /// let messages: Vec<f64> = vec![-3.2, 4.3, 0.12, -1.1, 2.78];
    ///
    /// // create a new Plaintext instance filled with the plaintexts we want
    /// let pt = Plaintext::encode(&messages, &encoder).unwrap();
    ///
    /// let n: usize = 2;
    /// let m = pt.decode_nth(n).unwrap();
    /// ```
    pub fn decode_nth(&self, nth: usize) -> Result<f64, CryptoAPIError> {
        if nth >= self.encoders.len() {
            return Err(IndexError!(self.nb_plaintexts, nth));
        }
        self.encoders[nth].decode_single(self.plaintexts[nth])
    }

    /// Encode several messages according to the list of Encoders in this instance
    /// # Arguments
    /// * `messages` - a list of messages as f64
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// // create a list of 5 Encoder instances where messages are in the interval [-5, 5[
    /// let encoders = vec![Encoder::new(-5., 5., 8, 0).unwrap(); 5];
    ///
    /// // create a list of messages in our interval
    /// let messages: Vec<f64> = vec![-3.2, 4.3, 0.12, -1.1, 2.78];
    ///
    /// // create a new Plaintext instance that can contain 5 plaintexts
    /// let mut ec = Plaintext::zero(5);
    ///
    /// // set the encoders
    /// ec.set_encoders(&encoders);
    ///
    /// // encode our messages
    /// ec.encode_inplace(&messages);
    /// ```
    pub fn encode_inplace(&mut self, messages: &[f64]) -> Result<(), CryptoAPIError> {
        debug_assert!(
            self.plaintexts.len() == self.encoders.len(),
            "self.plaintexts.len() != self.encoders.len()"
        );
        debug_assert!(
            self.plaintexts.len() == messages.len(),
            "self.plaintexts.len() != messages.len()"
        );
        for (pt, encoder, m) in izip!(
            self.plaintexts.iter_mut(),
            self.encoders.iter_mut(),
            messages.iter()
        ) {
            *pt = encoder.encode_core(*m)?;
        }
        Ok(())
    }

    /// Decode every plaintexts in this Plaintext instance according to its list of Encoders
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list of messages in our interval
    /// let messages: Vec<f64> = vec![-3.2, 4.3, 0.12, -1.1, 2.78];
    ///
    /// // create a new Plaintext instance filled with the plaintexts we want
    /// let mut ec = Plaintext::encode(&messages, &encoder).unwrap();
    ///
    /// let new_msgs: Vec<f64> = ec.decode().unwrap();
    /// ```
    pub fn decode(&self) -> Result<Vec<f64>, CryptoAPIError> {
        let mut result: Vec<f64> = vec![0.; self.nb_plaintexts];
        for (pt, encoder, r) in izip!(
            self.plaintexts.iter(),
            self.encoders.iter(),
            result.iter_mut()
        ) {
            *r = encoder.decode_core(*pt)?;
        }
        Ok(result)
    }

    /// Set the encoder list of this instance from an input list of encoders
    /// # Argument
    /// * `encoders` - a list of Encoder elements
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// let nb_ct = 100;
    /// let mut pt = Plaintext::zero(nb_ct);
    /// let encoders = vec![Encoder::zero(); nb_ct];
    /// // setting the encoders
    /// pt.set_encoders(&encoders);
    /// ```
    pub fn set_encoders(&mut self, encoders: &[crate::Encoder]) {
        debug_assert!(
            self.encoders.len() == encoders.len(),
            "self.encoders.len() != encoders.len()"
        );
        for (output, input) in izip!(self.encoders.iter_mut(), encoders.iter()) {
            output.copy(input);
        }
    }

    /// Set the encoder list of this instance from a unique input encoder
    /// # Argument
    /// * `encoder` - an Encoder
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// let nb_ct = 100;
    /// let mut pt = Plaintext::zero(nb_ct);
    /// let encoder = Encoder::zero();
    /// // setting the encoders
    /// pt.set_encoders_from_one(&encoder);
    /// ```
    pub fn set_encoders_from_one(&mut self, encoder: &crate::Encoder) {
        for output in self.encoders.iter_mut() {
            output.copy(encoder);
        }
    }

    /// Set the nth encoder of the encoder list of this instance from an input encoder
    /// # Argument
    /// * `encoder` - an Encoder
    /// # Example
    /// ```rust
    /// use concrete::{Encoder, Plaintext};
    ///
    /// let nb_ct = 100;
    /// let mut pt = Plaintext::zero(nb_ct);
    /// let encoder_1 = Encoder::zero();
    /// let encoder_2 = Encoder::zero();
    /// let n: usize = 2;
    /// // setting the encoders
    /// pt.set_encoders_from_one(&encoder_1);
    /// pt.set_nth_encoder(n, &encoder_2);
    /// ```
    pub fn set_nth_encoder(&mut self, nth: usize, encoder: &crate::Encoder) {
        self.encoders[nth].copy(encoder);
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<Plaintext, Box<dyn Error>> {
        read_from_file(path)
    }
}

/// Print needed pieces of information about this instance
impl fmt::Display for Plaintext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " Plaintext {\n";
        to_be_print =
            to_be_print + &format!("         -> nb_plaintexts = {}\n", self.nb_plaintexts);

        if self.nb_plaintexts <= 2 * n {
            for i in 0..self.nb_plaintexts {
                to_be_print = to_be_print
                    + &format!(
                        "         -> {} : value = {} | offset = {} | delta = {} | precision = {}\n",
                        i,
                        self.plaintexts[i],
                        self.encoders[i].o,
                        self.encoders[i].delta,
                        self.encoders[i].nb_bit_precision
                    );
            }
        } else {
            for i in 0..n {
                to_be_print = to_be_print
                    + &format!(
                        "         -> {} : value = {} | offset = {} | delta = {} | precision = {}\n",
                        i,
                        self.plaintexts[i],
                        self.encoders[i].o,
                        self.encoders[i].delta,
                        self.encoders[i].nb_bit_precision
                    );
            }
            to_be_print += "         -> ...\n";
            for i in (1..n + 1).rev() {
                to_be_print = to_be_print
                    + &format!(
                        "         -> {} : value = {} | offset = {} | delta = {} | precision = {}\n",
                        self.nb_plaintexts - i,
                        self.plaintexts[self.nb_plaintexts - i],
                        self.encoders[self.nb_plaintexts - i].o,
                        self.encoders[self.nb_plaintexts - i].delta,
                        self.encoders[self.nb_plaintexts - i].nb_bit_precision
                    );
            }
        }
        to_be_print += "      }";
        writeln!(f, "{}", to_be_print)
    }
}

#[cfg(test)]
mod tests;
