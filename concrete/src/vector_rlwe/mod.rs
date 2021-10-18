//! vector_rlwe ciphertext module

use std::error::Error;
use std::fmt;

use backtrace::Backtrace;
use colored::Colorize;
use itertools::izip;
use serde::{Deserialize, Serialize};
use concrete_core::{
    crypto::{encoding::PlaintextList, glwe::GlweList},
    math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor},
};
use concrete_npe as npe;
use crate::error::CryptoAPIError;
use crate::{read_from_file, write_to_file, Torus};
use concrete_commons::dispersion::StandardDev;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{CiphertextCount, GlweDimension, PolynomialSize};
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;

#[cfg(test)]
mod tests;

/// Structure containing a list of RLWE ciphertexts
/// They all have the same dimension (i.e. the length of the RLWE mask).
/// They all have the same number of coefficients in each of their polynomials (which is described by `polynomial_size`).
/// `polynomial_size` has to be a power of 2.
/// `nb_ciphertexts` has to be at least 1.
///
/// # Attributes
/// * `ciphertexts` - the concatenation of all the RLWE ciphertexts of the list
/// * `variances` - the variances of the noise of each RLWE ciphertext of the list
/// * `dimension` - the length the RLWE mask
/// * `polynomial_size` - the number of coefficients in a polynomial
/// * `nb_ciphertexts` - the number of RLWE ciphertexts present in the list
/// * `encoders` - the encoders of each RLWE ciphertext of the list
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorRLWE {
    pub ciphertexts: GlweList<Vec<Torus>>,
    pub variances: Vec<f64>,
    pub dimension: usize,
    pub polynomial_size: usize,
    pub nb_ciphertexts: usize,
    pub encoders: Vec<crate::Encoder>,
}

impl VectorRLWE {
    /// Instantiate a new VectorRLWE filled with zeros from a polynomial size, a dimension and a number of ciphertexts
    ///
    /// # Arguments
    /// * `polynomial_size` - the number of coefficients in polynomials
    /// * `dimension` - the length the RLWE mask
    /// * `nb_ciphertexts` - the number of RLWE ciphertexts to be stored in the structure
    ///
    /// # Output
    /// * a new instantiation of an VectorRLWE
    /// * NotPowerOfTwoError if `polynomial_size` is not a power of 2
    /// * ZeroCiphertextsInStructureError if we try to create a structure with no ciphertext in it
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // creates a list of 3 empty RLWE ciphertexts with a polynomial size of 630 and a dimension of 1
    /// let empty_ciphertexts = VectorRLWE::zero(1024, 1, 3).unwrap();
    /// ```
    pub fn zero(
        polynomial_size: usize,
        dimension: usize,
        nb_ciphertexts: usize,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        if nb_ciphertexts == 0 {
            return Err(ZeroCiphertextsInStructureError!(nb_ciphertexts));
        } else if (polynomial_size as f64
            - f64::powi(2., (polynomial_size as f64).log2().round() as i32))
        .abs()
            > f64::EPSILON
        {
            return Err(NotPowerOfTwoError!(polynomial_size));
        }

        Ok(VectorRLWE {
            ciphertexts: GlweList::allocate(
                0,
                PolynomialSize(polynomial_size),
                GlweDimension(dimension),
                CiphertextCount(nb_ciphertexts),
            ),
            variances: vec![0.; polynomial_size * nb_ciphertexts],
            dimension,
            polynomial_size,
            nb_ciphertexts,
            encoders: vec![crate::Encoder::zero(); nb_ciphertexts * polynomial_size],
        })
    }

    /// Encrypt several raw plaintexts (list of Torus element instead of a struct Plaintext) with the provided key and standard deviation into one RLWE ciphertext
    /// The slots of the RLWE ciphertext are filled with the provided plaintexts and if there are less plaintexts than slots, we pad with zeros
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintexts` - a list of plaintexts
    /// * `params` - RLWE parameters
    ///
    /// # Output
    /// * a new instantiation of an VectorRLWE encrypting the plaintexts provided in one ciphertext RLWE
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode
    /// let enc_messages = encoder.encode(&messages).unwrap();
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encrypt_packed(&sk, &enc_messages).unwrap();
    /// ```
    pub fn encrypt_packed(
        sk: &crate::RLWESecretKey,
        plaintexts: &crate::Plaintext,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        // compute the number of RLWE ct required to store all messages
        let nb_rlwe: usize =
            f64::ceil(plaintexts.nb_plaintexts as f64 / sk.polynomial_size as f64) as usize;

        // allocate the VectorRLWE structure
        let mut res = VectorRLWE::zero(sk.polynomial_size, sk.dimension, nb_rlwe)?;

        // create a vec with the good size for the raw plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; sk.polynomial_size * nb_rlwe];

        for (pt, pt_input, encoder_res, encoder_input, var_res) in izip!(
            tmp_pt.iter_mut(),
            plaintexts.plaintexts.iter(),
            res.encoders.iter_mut(),
            plaintexts.encoders.iter(),
            res.variances.iter_mut()
        ) {
            // encode messages
            *pt = *pt_input;

            // update the encoders
            encoder_res.copy(encoder_input);

            // update the variances
            *var_res = sk.get_variance();

            // check if there is an overlap of the noise into the message
            let nb_bit_overlap: usize = encoder_res.update_precision_from_variance(*var_res)?;

            // notification of a problem is there is overlap
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension to reduce the amount of noise needed.",
                    "Loss of precision during encrypt".red().bold(),
                    nb_bit_overlap, encoder_res.nb_bit_precision
                );
            }
        }

        // encrypt the plaintexts
        res.encrypt_packed_raw(sk, &tmp_pt)?;

        Ok(res)
    }

    /// Encode and encrypt several messages with the provided key into one RLWE ciphertext
    /// It means that the polynomial encrypted is P(X)=m0+m1X+m2X^2 ... with (m0, m1, m2, ...) the messages that have been encoded
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `messages` - a list of messages
    /// * `encoder` - an encoder
    ///
    /// # Output
    /// * a new instantiation of an VectorRLWE encrypting the plaintexts provided in one RLWE ciphertext
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode and encrypt
    /// let ct = VectorRLWE::encode_encrypt_packed(&sk, &messages, &encoder).unwrap();
    /// ```
    pub fn encode_encrypt_packed(
        sk: &crate::RLWESecretKey,
        messages: &[f64],
        encoder: &crate::Encoder,
    ) -> Result<VectorRLWE, CryptoAPIError> {
        // compute the number of RLWE ct required to store all messages
        let nb_rlwe: usize = f64::ceil(messages.len() as f64 / sk.polynomial_size as f64) as usize;

        // allocate the VectorRLWE structure
        let mut res = VectorRLWE::zero(sk.polynomial_size, sk.dimension, nb_rlwe)?;

        // create a vec with the good size for the raw plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; sk.polynomial_size * nb_rlwe];

        for (pt, m, encoder_res, var_res) in izip!(
            tmp_pt.iter_mut(),
            messages.iter(),
            res.encoders.iter_mut(),
            res.variances.iter_mut()
        ) {
            // encode messages
            *pt = encoder.encode_core(*m)?;

            // update the encoders
            encoder_res.copy(encoder);

            // update the variances
            *var_res = sk.get_variance();

            // check if there is an overlap of the noise into the message
            let nb_bit_overlap: usize = encoder_res.update_precision_from_variance(*var_res)?;

            // notification of a problem is there is overlap
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension to reduce the amount of noise needed.",
                    "Loss of precision during encrypt".red().bold(),
                    nb_bit_overlap, encoder_res.nb_bit_precision
                );
            }
        }

        // encrypt the plaintexts
        res.encrypt_packed_raw(sk, &tmp_pt)?;

        Ok(res)
    }

    /// Encode and encrypt n messages with the provided key into n RLWE ciphertext with only the constant coefficient filled with the message
    /// It means that the polynomial encrypted is P(X)=m with m the message that has been encoded
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintexts` - a list of plaintexts
    ///
    /// # Output
    /// * a new instantiation of an VectorRLWE encrypting the plaintexts provided in as many RLWE ciphertexts as there was messages
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode
    /// let enc_messages = encoder.encode(&messages).unwrap();
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encrypt(&sk, &enc_messages).unwrap();
    /// ```
    pub fn encrypt(
        sk: &crate::RLWESecretKey,
        plaintexts: &crate::Plaintext,
    ) -> Result<VectorRLWE, CryptoAPIError> {
        // get the number of ciphertexts to output
        let nb_ciphertexts: usize = plaintexts.nb_plaintexts;

        // allocate the VectorRLWE structure
        let mut res = VectorRLWE::zero(sk.polynomial_size, sk.dimension, nb_ciphertexts)?;

        // create a vec with the good size for the raw plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; sk.polynomial_size * nb_ciphertexts];

        for (pt, pt_input, encoder_res, encoder_input, var_res) in izip!(
            tmp_pt.chunks_mut(sk.polynomial_size),
            plaintexts.plaintexts.iter(),
            res.encoders.chunks_mut(sk.polynomial_size),
            plaintexts.encoders.iter(),
            res.variances.chunks_mut(sk.polynomial_size)
        ) {
            // encode messages
            pt[0] = *pt_input;

            // update the encoders
            encoder_res[0].copy(encoder_input);

            // update the variances
            var_res[0] = sk.get_variance();

            // check if there is an overlap of the noise into the message
            let nb_bit_overlap: usize =
                encoder_res[0].update_precision_from_variance(var_res[0])?;

            // notification of a problem is there is overlap
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension to reduce the amount of noise needed.",
                    "Loss of precision during encrypt".red().bold(),
                    nb_bit_overlap, encoder_res[0].nb_bit_precision
                );
            }
        }

        // encrypt the plaintexts
        res.encrypt_packed_raw(sk, &tmp_pt)?;

        Ok(res)
    }

    /// Encode and encrypt n messages with the provided key into n RLWE ciphertext with only the constant coefficient filled with the message
    /// It means that the polynomial encrypted is P(X)=m with m the the message that have been encoded
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `messages` - a list of messages
    /// * `encoder` - an encoder
    ///
    /// # Output
    /// * a new instantiation of an VectorRLWE encrypting the plaintexts provided in as many RLWE ciphertexts as there was messages
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encode_encrypt(&sk, &messages, &encoder).unwrap();
    /// ```
    pub fn encode_encrypt(
        sk: &crate::RLWESecretKey,
        messages: &[f64],
        encoder: &crate::Encoder,
    ) -> Result<VectorRLWE, CryptoAPIError> {
        // get the number of ciphertexts to output
        let nb_ciphertexts: usize = messages.len();

        // allocate the VectorRLWE structure
        let mut res = VectorRLWE::zero(sk.polynomial_size, sk.dimension, nb_ciphertexts)?;

        // create a vec with the good size for the raw plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; sk.polynomial_size * nb_ciphertexts];

        for (pt, m, encoder_res, var_res) in izip!(
            tmp_pt.chunks_mut(sk.polynomial_size),
            messages.iter(),
            res.encoders.chunks_mut(sk.polynomial_size),
            res.variances.chunks_mut(sk.polynomial_size)
        ) {
            // encode messages
            pt[0] = encoder.encode_core(*m)?;

            // update the encoders
            encoder_res[0].copy(encoder);

            // update the variances
            var_res[0] = sk.get_variance();

            // check if there is an overlap of the noise into the message
            let nb_bit_overlap: usize =
                encoder_res[0].update_precision_from_variance(var_res[0])?;

            // notification of a problem is there is overlap
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension to reduce the amount of noise needed.",
                    "Loss of precision during encrypt".red().bold(),
                    nb_bit_overlap, encoder_res[0].nb_bit_precision
                );
            }
        }

        // encrypt the plaintexts
        res.encrypt_packed_raw(sk, &tmp_pt)?;

        Ok(res)
    }

    /// Encrypt several raw plaintexts (list of Torus element instead of a struct Plaintext) with the provided key and standard deviation into several ciphertexts RLWE (each coefficient of the polynomial plaintexts is filled)
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintexts` - a list of plaintexts
    ///
    /// # Output
    /// * WrongSizeError if the plaintext slice length is not a multiple of polynomial size
    /// * NoNoiseInCiphertextError if the noise distribution is too small for the integer representation
    pub fn encrypt_packed_raw(
        &mut self,
        sk: &crate::RLWESecretKey,
        plaintexts: &[Torus],
    ) -> Result<(), CryptoAPIError> {
        // the plaintext slice length should be a multiple of polynomial size
        if plaintexts.len() % self.polynomial_size != 0 {
            return Err(WrongSizeError!(plaintexts.len()));
        }
        // check if we have enough std dev to have noise in the ciphertext
        else if sk.std_dev < f64::powi(2., -(<Torus as Numeric>::BITS as i32) + 2) {
            return Err(NoNoiseInCiphertext!(sk.get_variance()));
        }

        // set the variances
        self.variances = vec![sk.get_variance(); self.nb_ciphertexts * self.polynomial_size];

        sk.val.encrypt_glwe_list(
            &mut self.ciphertexts,
            &PlaintextList::from_container(plaintexts),
            StandardDev::from_standard_dev(sk.std_dev),
            &mut EncryptionRandomGenerator::new(None),
        );

        Ok(())
    }

    /// Compute the decryption of each ciphertext
    ///
    /// # Argument
    /// * `sk` - an glwe secret key
    ///
    /// # Output
    /// * an array of f64
    /// * PolynomialSizeError - if the polynomial size of the secret key and the polynomial size of the RLWE ciphertext are different
    /// * DimensionError - if the dimension of the secret key and the dimension of the RLWE cipertext are different
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode
    /// let enc_messages = encoder.encode(&messages).unwrap();
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encrypt(&sk, &enc_messages).unwrap();
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode(&self, sk: &crate::RLWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
        if sk.polynomial_size != self.polynomial_size {
            return Err(PolynomialSizeError!(
                sk.polynomial_size,
                self.polynomial_size
            ));
        } else if sk.dimension != self.dimension {
            return Err(DimensionError!(sk.dimension, self.dimension));
        }

        let mut result: Vec<f64> = vec![0.; self.nb_valid()];

        // create a vec with the good size for the plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; self.polynomial_size * self.nb_ciphertexts];

        // compute the phase for all ciphertext
        sk.val.decrypt_glwe_list(
            &mut PlaintextList::from_container(tmp_pt.as_mut_slice()),
            &self.ciphertexts,
        );

        // decode as soon as the encoding is valid
        let mut i: usize = 0;
        for (pt, encoder) in izip!(tmp_pt.iter(), self.encoders.iter()) {
            if encoder.is_valid() {
                result[i] = encoder.decode_single(*pt)?;
                i += 1;
            }
        }
        Ok(result)
    }

    /// Compute the decryption of each ciphertext in a rounding setting
    ///
    /// # Argument
    /// * `sk` - an glwe secret key
    ///
    /// # Output
    /// * an array of f64
    /// * PolynomialSizeError - if the polynomial size of the secret key and the polynomial size of the RLWE ciphertext are different
    /// * DimensionError - if the dimension of the secret key and the dimension of the RLWE cipertext are different
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode
    /// let enc_messages = encoder.encode(&messages).unwrap();
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encrypt(&sk, &enc_messages).unwrap();
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode_round(
        &self,
        sk: &crate::RLWESecretKey,
    ) -> Result<Vec<f64>, CryptoAPIError> {
        if sk.polynomial_size != self.polynomial_size {
            return Err(PolynomialSizeError!(
                sk.polynomial_size,
                self.polynomial_size
            ));
        } else if sk.dimension != self.dimension {
            return Err(DimensionError!(sk.dimension, self.dimension));
        }

        let mut result: Vec<f64> = vec![0.; self.nb_valid()];

        // create a vec with the good size for the plaintexts
        let mut tmp_pt: Vec<Torus> = vec![0; self.polynomial_size * self.nb_ciphertexts];

        // compute the phase for all ciphertext
        sk.val.decrypt_glwe_list(
            &mut PlaintextList::from_container(tmp_pt.as_mut_slice()),
            &self.ciphertexts,
        );

        // decode as soon as the encoding is valid
        let mut i: usize = 0;
        for (pt, encoder) in izip!(tmp_pt.iter(), self.encoders.iter()) {
            if encoder.is_valid() {
                let mut encoder_round = encoder.clone();
                encoder_round.round = true;
                result[i] = encoder_round.decode_single(*pt)?;
                i += 1;
            }
        }
        Ok(result)
    }

    /// Compute the decryption of each ciphertext and returns also the associated encoder
    /// if nb=3 we return the coefficient 0 of the ciphertext 0,
    /// the coefficient 1 of the ciphertext 0 and the coefficient 2 of the ciphertext 0
    ///
    /// # Argument
    /// * `sk` - an glwe secret key
    /// * `nb` - the number of coeff we want to decrypt
    ///
    /// # Output
    /// * an array of f64
    /// * an array of encoders
    /// * PolynomialSizeError - if the polynomial size of the secret key and the polynomial size of the RLWE ciphertext are different
    /// * DimensionError - if the dimension of the secret key and the dimension of the RLWE cipertext are different
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode
    /// let enc_messages = encoder.encode(&messages).unwrap();
    ///
    /// // encrypt and decrypt
    /// let ct = VectorRLWE::encrypt(&sk, &enc_messages).unwrap();
    /// let (res, encoders) = ct.decrypt_with_encoders(&sk).unwrap();
    /// ```
    pub fn decrypt_with_encoders(
        &self,
        sk: &crate::RLWESecretKey,
    ) -> Result<(Vec<f64>, Vec<crate::Encoder>), CryptoAPIError> {
        let messages = self.decrypt_decode(sk)?;
        let mut encoders: Vec<crate::Encoder> = vec![crate::Encoder::zero(); messages.len()];
        let mut cpt: usize = 0;
        for e in self.encoders.iter() {
            if e.is_valid() {
                encoders[cpt].copy(e);
                cpt += 1;
            }
        }
        Ok((messages, encoders))
    }

    /// Extract the n_coeff-th coefficient of the n_ciphertext-th RLWE ciphertext
    ///
    /// # Argument
    /// * `n_coeff` - the desired coefficient, starts at zero
    /// * `n_ciphertext` - the desired RLWE ciphertext, starts at zero
    ///
    /// # Output
    /// * the desired LWE as a VectorRLWE structure
    /// * IndexError - if the requested ciphertext does not exist
    /// * MonomialError - if the requested monomial does not exist
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // generate a secret key
    /// let dimension: usize = 1;
    /// let polynomial_size: usize = 1024;
    /// let log_std_dev: i32 = -20;
    /// let sk = RLWESecretKey::new(&RLWE128_1024_1);
    ///
    /// // random settings for the encoder and some random messages
    /// let (min, max) = (-43., -10.);
    /// let (precision, padding) = (5, 2);
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    /// let messages: Vec<f64> = vec![-39.69, -19.37, -40.74, -41.26, -35.77];
    ///
    /// // encode and encrypt
    /// let ct = VectorRLWE::encode_encrypt_packed(&sk, &messages, &encoder).unwrap();
    ///
    /// // convert into LWE secret key
    /// let lwe_sk = sk.to_lwe_secret_key();
    ///
    /// // extract a filled coefficient
    /// let n_coeff = 2;
    /// let n_ct = 0;
    /// let res = ct.extract_1_lwe(n_coeff, n_ct).unwrap();
    /// ```
    pub fn extract_1_lwe(
        &self,
        n_coeff: usize,
        n_ciphertext: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        // ciphertext index too big
        if n_ciphertext > self.nb_ciphertexts - 1 {
            return Err(IndexError!(self.nb_ciphertexts, n_ciphertext));
        }
        // monomial index too big
        else if n_coeff > self.polynomial_size - 1 {
            return Err(MonomialError!(self.polynomial_size, n_coeff));
        }

        // compute the dimension of the output LWE
        let lwe_dimension = self.dimension * self.polynomial_size;

        // allocation for the result
        let mut res = crate::VectorLWE::zero(lwe_dimension, 1)?;

        // compute the index for the variance and the encoder
        let index = n_coeff + n_ciphertext * self.polynomial_size;

        // fill the variance and the encoder
        res.variances[0] = self.variances[index];
        res.encoders[0].copy(&self.encoders[index]);

        // compute the index for the body
        let index_body: usize = n_ciphertext * self.polynomial_size * (self.dimension + 1)
            + self.dimension * self.polynomial_size
            + n_coeff;

        // fill the body
        *res.ciphertexts
            .as_mut_tensor()
            .get_element_mut(lwe_dimension) = *self.ciphertexts.as_tensor().get_element(index_body);

        // fill the mask
        res.ciphertexts
            .as_mut_tensor()
            .as_mut_slice()
            .get_mut(0..(self.dimension * self.polynomial_size))
            .unwrap()
            .copy_from_slice(
                self.ciphertexts
                    .as_tensor()
                    .as_slice()
                    .get(
                        (n_ciphertext * self.polynomial_size * (self.dimension + 1))
                            ..(n_ciphertext * self.polynomial_size * (self.dimension + 1)
                                + self.dimension * self.polynomial_size),
                    )
                    .unwrap(),
            );

        // deal with the rotation and the signs
        let rot: usize = self.polynomial_size - n_coeff - 1;

        for mut polynomial in res
            .ciphertexts
            .as_mut_tensor()
            .get_sub_mut(0..(self.dimension * self.polynomial_size))
            .subtensor_iter_mut(self.polynomial_size)
        {
            if polynomial.len() == self.polynomial_size {
                polynomial.as_mut_slice().reverse();
                for elt in polynomial
                    .as_mut_slice()
                    .get_mut(0..rot)
                    .unwrap()
                    .iter_mut()
                {
                    *elt = elt.wrapping_neg();
                }
                polynomial.as_mut_slice().rotate_left(rot);
            }
        }

        Ok(res)
    }

    /// Add small messages to a VectorRLWE ciphertext and does not change the encoding but changes the bodies of the ciphertexts
    /// the first message is added to the first coefficient that has a valid encoder
    /// the second message is added to the second coefficient that has a valid encoder
    /// ...
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Output
    /// * A new VectorRLWE
    /// * NotEnoughValidEncoderError - if messages is bigger than the number of valid encoders
    pub fn add_constant_static_encoder(
        &self,
        messages: &[f64],
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_static_encoder_inplace(messages)?;
        Ok(res)
    }

    /// Add small messages to a VectorRLWE ciphertext and does not change the encoding but changes the bodies of the ciphertexts
    /// the first message is added to the first coefficient that has a valid encoder
    /// the second message is added to the second coefficient that has a valid encoder
    /// ...
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Output
    /// * NotEnoughValidEncoderError - if messages is bigger than the number of valid encoders
    pub fn add_constant_static_encoder_inplace(
        &mut self,
        messages: &[f64],
    ) -> Result<(), CryptoAPIError> {
        let nb_valid = self.nb_valid();
        if nb_valid < messages.len() {
            return Err(NotEnoughValidEncoderError!(nb_valid, messages.len()));
        }

        // counter for messages
        let mut cpt: usize = 0;

        // loop over RLWE ciphertexts
        for (mut ciphertext, encoders) in izip!(
            self.ciphertexts
                .as_mut_tensor()
                .subtensor_iter_mut((self.dimension + 1) * self.polynomial_size),
            self.encoders.chunks_mut(self.polynomial_size)
        ) {
            for (monomial_coeff, encoder) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut(
                        (self.polynomial_size * self.dimension)
                            ..(self.polynomial_size * (self.dimension + 1))
                    )
                    .unwrap()
                    .iter_mut(),
                encoders.iter_mut()
            ) {
                if encoder.is_valid() {
                    // select the next message
                    let m = messages[cpt];

                    // error if one message is not in [-delta,delta]
                    if m.abs() > encoder.delta {
                        return Err(MessageTooBigError!(m, encoder.delta));
                    }

                    let mut ec_tmp = encoder.clone();
                    ec_tmp.o = 0.;
                    *monomial_coeff =
                        monomial_coeff.wrapping_add(ec_tmp.encode_outside_interval_operators(m)?);

                    cpt += 1;
                }
            }
        }

        Ok(())
    }

    /// Add messages to an VectorRLWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertexts
    /// the first message is added to the first coefficient that has a valid encoder
    /// the second message is added to the second coefficient that has a valid encoder
    /// ...
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Output
    /// * a new VectorRLWE
    /// * NotEnoughValidEncoderError - if messages is bigger than the number of valid encoders
    pub fn add_constant_dynamic_encoder(
        &self,
        messages: &[f64],
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_dynamic_encoder_inplace(messages)?;
        Ok(res)
    }

    /// Add messages to an VectorRLWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertexts
    /// the first message is added to the first coefficient that has a valid encoder
    /// the second message is added to the second coefficient that has a valid encoder
    /// ...
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    /// * NotEnoughValidEncoderError - if messages is bigger than the number of valid encoders
    pub fn add_constant_dynamic_encoder_inplace(
        &mut self,
        messages: &[f64],
    ) -> Result<(), CryptoAPIError> {
        let nb_valid = self.nb_valid();
        if nb_valid < messages.len() {
            return Err(NotEnoughValidEncoderError!(nb_valid, messages.len()));
        }

        // counter for messages
        let mut cpt: usize = 0;

        // add the message
        for lwe_encoder in self.encoders.iter_mut() {
            if lwe_encoder.is_valid() {
                lwe_encoder.o += messages[cpt];
                cpt += 1;
            }
        }

        Ok(())
    }

    /// Compute an homomorphic addition between two VectorRLWE ciphertexts and the center of the output Encoder is the sum of the two centers of the input Encoders
    ///
    /// # Arguments
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * a new VectorRLWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * DeltaError - if the ciphertext encoders have incompatible deltas
    pub fn add_centered(
        &self,
        ct: &crate::VectorRLWE,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_centered_inplace(ct)?;
        Ok(res)
    }

    /// Compute an homomorphic addition between two VectorRLWE ciphertexts and the center of the output Encoder is the sum of the two centers of the input Encoders
    ///
    /// # Arguments
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * DeltaError - if the ciphertext encoders have incompatible deltas
    pub fn add_centered_inplace(&mut self, ct: &crate::VectorRLWE) -> Result<(), CryptoAPIError> {
        // check same dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }
        // check same polynomial sizes
        else if self.polynomial_size != ct.polynomial_size {
            return Err(PolynomialSizeError!(
                self.polynomial_size,
                ct.polynomial_size
            ));
        }

        // check same deltas
        for (self_enc, ct_enc) in self.encoders.iter_mut().zip(ct.encoders.iter()) {
            if self_enc.is_valid() && ct_enc.is_valid() && !deltas_eq!(self_enc.delta, ct_enc.delta)
            {
                return Err(DeltaError!(self_enc.delta, ct_enc.delta));
            }
        }

        // add ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_add(ct.ciphertexts.as_tensor());

        // correction related to the addition
        for (mut ciphertext, encoders, encoders_ct, self_variances, ct_variances) in izip!(
            self.ciphertexts
                .as_mut_tensor()
                .subtensor_iter_mut(self.polynomial_size * (self.dimension + 1)),
            self.encoders.chunks_mut(self.polynomial_size),
            ct.encoders.chunks(self.polynomial_size),
            self.variances.chunks_mut(self.polynomial_size),
            ct.variances.chunks(self.polynomial_size),
        ) {
            for (monomial_coeff, encoder, encoder_ct, self_var, ct_var) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut(
                        (self.polynomial_size * self.dimension)
                            ..(self.polynomial_size * (self.dimension + 1))
                    )
                    .unwrap()
                    .iter_mut(),
                encoders.iter_mut(),
                encoders_ct.iter(),
                self_variances.iter_mut(),
                ct_variances.iter()
            ) {
                // compute the new variance
                *self_var = npe::add_ciphertexts(*self_var, *ct_var);

                // both coefficients contained a message
                if encoder.is_valid() && encoder_ct.is_valid() {
                    let mut tmp_enc = encoder.clone();
                    tmp_enc.o = 0.;
                    let correction: Torus = tmp_enc.encode_core(encoder.delta / 2.)?;
                    *monomial_coeff = monomial_coeff.wrapping_sub(correction);
                    encoder.o += encoder_ct.o + encoder.delta / 2.;

                    // update the encoder precision based on the variance
                    encoder.update_precision_from_variance(*self_var)?;
                }
                // only the ct coefficient contained a message
                else if !encoder.is_valid() && encoder_ct.is_valid() {
                    encoder.copy(encoder_ct);

                    // update the encoder precision based on the variance
                    encoder.update_precision_from_variance(*self_var)?;
                }
            }
        }
        Ok(())
    }

    /// Compute an addition between two VectorRLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * PaddingError - if the ciphertexts ave incompatible paddings
    /// * NotEnoughPaddingError - if there is no padding
    /// * DeltaError - if the ciphertexts have incompatile deltas
    pub fn add_with_padding(
        &self,
        ct: &crate::VectorRLWE,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an addition between two VectorRLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * a new VectorRLWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * PaddingError - if the ciphertexts ave incompatible paddings
    /// * NotEnoughPaddingError - if there is no padding
    /// * DeltaError - if the ciphertexts have incompatile deltas
    pub fn add_with_padding_inplace(
        &mut self,
        ct: &crate::VectorRLWE,
    ) -> Result<(), CryptoAPIError> {
        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }
        // check the polynomial sizes
        else if self.polynomial_size != ct.polynomial_size {
            return Err(PolynomialSizeError!(
                self.polynomial_size,
                ct.polynomial_size
            ));
        }

        // check the Encoder lists
        for (self_enc, ct_enc) in self.encoders.iter_mut().zip(ct.encoders.iter()) {
            if self_enc.is_valid() && ct_enc.is_valid() {
                // check nb bit padding
                if self_enc.nb_bit_padding != ct_enc.nb_bit_padding {
                    return Err(PaddingError!(
                        self_enc.nb_bit_padding,
                        ct_enc.nb_bit_padding
                    ));
                }
                // check the paddings
                else if self_enc.nb_bit_padding == 0 {
                    return Err(NotEnoughPaddingError!(self_enc.nb_bit_padding, 1));
                }
                // check the deltas
                else if !deltas_eq!(self_enc.delta, ct_enc.delta) {
                    return Err(DeltaError!(self_enc.delta, ct_enc.delta));
                }
            }
        }

        // add the ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_add(ct.ciphertexts.as_tensor());

        // update the Encoder list
        for (self_enc, ct_enc, self_var, ct_var) in izip!(
            self.encoders.iter_mut(),
            ct.encoders.iter(),
            self.variances.iter_mut(),
            ct.variances.iter()
        ) {
            // compute the new variance
            *self_var = npe::add_ciphertexts(*self_var, *ct_var);

            // compute the new encoder
            if self_enc.is_valid() && ct_enc.is_valid() {
                self_enc.o += ct_enc.o;
                self_enc.delta *= 2.;
                self_enc.nb_bit_padding -= 1;
            } else if !self_enc.is_valid() && ct_enc.is_valid() {
                self_enc.copy(ct_enc);
            }

            if self_enc.is_valid() {
                // update the encoder precision based on the variance
                self_enc.update_precision_from_variance(*self_var)?;
            }
        }

        Ok(())
    }

    /// Compute an addition between two VectorRLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * a new VectorRLWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * PaddingError - if the ciphertexts ave incompatible paddings
    /// * NotEnoughPaddingError - if there is no padding
    /// * DeltaError - if the ciphertexts have incompatile deltas
    /// * InvalidEncoderError - if one of the ciphertext have an invalid encoder
    pub fn sub_with_padding(
        &self,
        ct: &crate::VectorRLWE,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.sub_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an addition between two VectorRLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorRLWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * PolynomialSizeError - if the ciphertexts have incompatible polynomial size
    /// * PaddingError - if the ciphertexts ave incompatible paddings
    /// * NotEnoughPaddingError - if there is no padding
    /// * DeltaError - if the ciphertexts have incompatile deltas
    pub fn sub_with_padding_inplace(
        &mut self,
        ct: &crate::VectorRLWE,
    ) -> Result<(), CryptoAPIError> {
        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }
        // check the polynomial sizes
        else if self.polynomial_size != ct.polynomial_size {
            return Err(PolynomialSizeError!(
                self.polynomial_size,
                ct.polynomial_size
            ));
        }
        // check the encoder lists
        for (self_enc, ct_enc) in self.encoders.iter_mut().zip(ct.encoders.iter()) {
            if self_enc.is_valid() && ct_enc.is_valid() {
                // check the paddings
                if self_enc.nb_bit_padding != ct_enc.nb_bit_padding {
                    return Err(PaddingError!(
                        self_enc.nb_bit_padding,
                        ct_enc.nb_bit_padding
                    ));
                }
                // check enough padding
                else if self_enc.nb_bit_padding == 0 {
                    return Err(NotEnoughPaddingError!(self_enc.nb_bit_padding, 1));
                }
                // check deltas
                else if !deltas_eq!(self_enc.delta, ct_enc.delta) {
                    return Err(DeltaError!(self_enc.delta, ct_enc.delta));
                }
            }
        }

        // subtract ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_sub(ct.ciphertexts.as_tensor());

        // correction related to the subtraction
        for (mut ciphertext, enc1_list, enc2_list) in izip!(
            self.ciphertexts
                .as_mut_tensor()
                .subtensor_iter_mut((self.dimension + 1) * self.polynomial_size),
            self.encoders.chunks(self.polynomial_size),
            ct.encoders.chunks(self.polynomial_size),
        ) {
            for (coeff, enc1, enc2) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut(
                        (self.dimension * self.polynomial_size)
                            ..((self.dimension + 1) * self.polynomial_size),
                    )
                    .unwrap()
                    .iter_mut(),
                enc1_list.iter(),
                enc2_list.iter()
            ) {
                if enc1.is_valid() && enc2.is_valid() {
                    let correction: Torus = 1 << (<Torus as Numeric>::BITS - enc1.nb_bit_padding);
                    *coeff = coeff.wrapping_add(correction);
                }
            }
        }

        // update the Encoder list
        for (self_enc, ct_enc, self_var, ct_var) in izip!(
            self.encoders.iter_mut(),
            ct.encoders.iter(),
            self.variances.iter_mut(),
            ct.variances.iter()
        ) {
            // compute the new variance
            *self_var = npe::add_ciphertexts(*self_var, *ct_var);

            // compute the new encoder
            if self_enc.is_valid() && ct_enc.is_valid() {
                self_enc.o -= ct_enc.o + ct_enc.delta;
                self_enc.delta *= 2.;
                self_enc.nb_bit_padding -= 1;
            } else if !self_enc.is_valid() && ct_enc.is_valid() {
                self_enc.copy(ct_enc);
            }

            if self_enc.is_valid() {
                // update the encoder precision based on the variance
                self_enc.update_precision_from_variance(*self_var)?;
            }
        }

        Ok(())
    }

    /// Multiply VectorRLWE ciphertexts with small integer messages and does not change the encoding but changes the bodies and masks of the ciphertexts
    /// # Argument
    /// * `messages` - a list of integer messages as Torus elements
    pub fn mul_constant_static_encoder_inplace(
        &mut self,
        messages: &[i32],
    ) -> Result<(), CryptoAPIError> {
        for (mut ciphertext, m, encoder_list, variance_list) in izip!(
            self.ciphertexts
                .as_mut_tensor()
                .subtensor_iter_mut((self.dimension + 1) * self.polynomial_size),
            messages.iter(),
            self.encoders.chunks_mut(self.polynomial_size),
            self.variances.chunks_mut(self.polynomial_size),
        ) {
            // compute the multiplication
            ciphertext.update_with_wrapping_scalar_mul(&(*m as Torus));

            // compute the absolute value
            let m_abs = m.abs();

            for (coeff, enc, var) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut(
                        ((self.dimension) * self.polynomial_size)
                            ..((self.dimension + 1) * self.polynomial_size)
                    )
                    .unwrap()
                    .iter_mut(),
                encoder_list.iter_mut(),
                variance_list.iter_mut(),
            ) {
                if enc.is_valid() {
                    // compute correction
                    let cor0: Torus = enc.encode_outside_interval_operators(0.)?;
                    let cor = cor0.wrapping_mul((*m - 1) as Torus);

                    // apply correction
                    *coeff = coeff.wrapping_sub(cor);
                }
                // call to the NPE to estimate the new variance
                *var = npe::LWE::single_scalar_mul(*var, m_abs as Torus);

                if m_abs != 0 {
                    // update the encoder precision based on the variance
                    enc.update_precision_from_variance(*var)?;
                }
            }
        }
        Ok(())
    }

    /// Multiply each VectorRLWE ciphertext with a real constant and do change the encoding and the ciphertexts by consuming some bits of padding
    /// it needs to have the same number of constant than ciphertexts
    /// it also needs that the input encoding all contained zero in their intervals
    /// the output precision is the minimum between the input and the number of bits of padding consumed
    ///
    /// # Argument
    /// * `scale` - a positive scaling factor which has to be greater that any of the messages.abs()
    /// * `nb_bit_padding` - the number of bits of padding to be consumed
    /// * `messages` - a list of real messages as f64
    ///
    /// # Output
    pub fn mul_constant_with_padding(
        &self,
        constants: &[f64],
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<crate::VectorRLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.mul_constant_with_padding_inplace(constants, max_constant, nb_bit_padding)?;
        Ok(res)
    }

    /// Multiply each VectorRLWE ciphertext with a real constant and do change the encoding and the ciphertexts by consuming some bits of padding
    /// it needs to have the same number of constant than ciphertexts
    /// it also needs that the input encoding all contained zero in their intervals
    /// the output precision is the minimum between the input and the number of bits of padding consumed
    ///
    /// # Argument
    /// * `scale` - a positive scaling factor which has to be greater that any of the messages.abs()
    /// * `nb_bit_padding` - the number of bits of padding to be consumed
    /// * `messages` - a list of real messages as f64
    ///
    /// # Output
    /// * NbCTError - if the ciphertext and the constants have not the same number of samples
    /// * ConstantMaximumError - if the absolute value of a coefficient in `constants` is bigger than `max_constant`
    /// * ZeroInIntervalError - if 0 is not in the encoder interval
    /// * NotEnoughPaddingError - if there is not enough padding for the operation
    pub fn mul_constant_with_padding_inplace(
        &mut self,
        constants: &[f64],
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<(), CryptoAPIError> {
        // check if we have the same number of messages and ciphertexts
        if constants.len() != self.nb_valid() {
            return Err(NbCTError!(constants.len(), self.nb_valid()));
        }

        // check that the constant if below the maximum
        for c in constants.iter() {
            if *c > max_constant || *c < -max_constant {
                return Err(ConstantMaximumError!(*c, max_constant));
            }
        }

        // some checks about the valid encoders
        for encoder in self.encoders.iter_mut() {
            if encoder.is_valid() {
                // check that zero is in the interval
                if encoder.o > 0. || encoder.o + encoder.delta < 0. {
                    return Err(ZeroInIntervalError!(encoder.o, encoder.delta));
                }
                // check bits of paddings
                else if encoder.nb_bit_padding < nb_bit_padding {
                    return Err(NotEnoughPaddingError!(
                        encoder.nb_bit_padding,
                        nb_bit_padding
                    ));
                }
            }
        }

        // store the size of one RLWE
        let ct_size: usize = self.get_ciphertext_size();

        for (mut ciphertext, c, encoders, variances) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            constants.iter(),
            self.encoders.chunks_mut(self.polynomial_size),
            self.variances.chunks_mut(self.polynomial_size),
        ) {
            // test if negative
            let negative: bool = *c < 0.;

            // absolute value
            let c_abs = c.abs();

            // discretize c_abs with regard to the number of bits of padding to use
            let scal: Torus =
                (c_abs / max_constant * f64::powi(2., nb_bit_padding as i32)).round() as Torus;

            // subtract the encoded zeros (pre mul correction)
            for (b, encoder) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut((self.polynomial_size * self.dimension)..(ct_size))
                    .unwrap()
                    .iter_mut(),
                encoders.iter()
            ) {
                if encoder.is_valid() {
                    *b = b.wrapping_sub(encoder.encode_core(0.)?);
                }
            }

            // scalar multiplication
            ciphertext.update_with_wrapping_scalar_mul(&scal);

            // compute the discretization of c_abs
            let discret_c_abs =
                (scal as f64) * f64::powi(2., -(nb_bit_padding as i32)) * max_constant;

            // compute  the rounding error on c_abs
            let rounding_error = (discret_c_abs - c_abs).abs();

            // post mul correction and new encoders
            for (b, encoder, var) in izip!(
                ciphertext
                    .as_mut_slice()
                    .get_mut((self.polynomial_size * self.dimension)..(ct_size))
                    .unwrap()
                    .iter_mut(),
                encoders.iter_mut(),
                variances.iter_mut()
            ) {
                // call to the NPE to estimate the new variance
                *var = npe::LWE::single_scalar_mul(*var, scal);
                if scal != 0 {
                    // update the encoder precision based on the variance
                    encoder.update_precision_from_variance(*var)?;
                }

                if encoder.is_valid() {
                    // new encoder
                    let new_o = encoder.o * max_constant;
                    let new_max =
                        (encoder.o + encoder.delta - encoder.get_granularity()) * max_constant;
                    let new_delta = new_max - new_o;

                    // get the ciphertext granularity
                    let granularity = encoder.get_granularity();

                    // compute the max of the ciphertext (based on the metadata of the encoder)
                    let max = f64::max(
                        (encoder.o + encoder.delta - encoder.get_granularity()).abs(),
                        encoder.o.abs(),
                    );
                    // compute the new granularity
                    let new_granularity = 2.
                        * (granularity * rounding_error / 2.
                            + granularity / 2. * discret_c_abs
                            + rounding_error * max)
                            .abs();

                    // compute the new precision
                    let new_precision = usize::min(
                        f64::log2(new_delta / new_granularity).floor() as usize,
                        encoder.nb_bit_precision,
                    );

                    // create the new encoder
                    let tmp_encoder = crate::Encoder::new(
                        new_o,
                        new_max,
                        usize::min(nb_bit_padding, encoder.nb_bit_precision),
                        encoder.nb_bit_padding - nb_bit_padding,
                    )?;
                    encoder.copy(&tmp_encoder);
                    encoder.nb_bit_precision = usize::min(encoder.nb_bit_precision, new_precision);

                    // encode 0 with the new encoder
                    let tmp_add = encoder.encode_core(0.)?;
                    *b = b.wrapping_add(tmp_add);
                }
            }

            if negative {
                // compute the opposite
                ciphertext.update_with_wrapping_neg();

                for (b, encoder) in izip!(
                    ciphertext
                        .as_mut_slice()
                        .get_mut((self.polynomial_size * self.dimension)..(ct_size))
                        .unwrap()
                        .iter_mut(),
                    encoders.iter_mut()
                ) {
                    if encoder.is_valid() {
                        // add correction if there is some padding
                        let mut cor: Torus = 0;
                        if encoder.nb_bit_padding > 0 {
                            cor = (1 << (<Torus as Numeric>::BITS - encoder.nb_bit_padding))
                                - (1 << (<Torus as Numeric>::BITS
                                    - encoder.nb_bit_padding
                                    - encoder.nb_bit_precision));
                        } else {
                            cor = cor.wrapping_sub(
                                1 << (<Torus as Numeric>::BITS
                                    - encoder.nb_bit_padding
                                    - encoder.nb_bit_precision),
                            );
                        }
                        *b = b.wrapping_add(cor);

                        // change the encoder
                        encoder.opposite_inplace()?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Return the number of valid encoders (i.e. how many messages are carried in those RLWE ciphertexts)
    pub fn nb_valid(&self) -> usize {
        let mut res: usize = 0;
        for enc in self.encoders.iter() {
            if enc.is_valid() {
                res += 1;
            }
        }
        res
    }

    pub fn get_ciphertext_size(&self) -> usize {
        self.polynomial_size * (self.dimension + 1)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<VectorRLWE, Box<dyn Error>> {
        read_from_file(path)
    }
}

// pub ciphertexts: Vec<Torus>,
// pub variances: Vec<f64>,
// pub dimension: usize,
// pub polynomial_size: usize,
// pub nb_ciphertexts: usize,
// pub encoders: Vec<Encoder>,
/// Print needed pieces of information about an VectorRLWE
impl fmt::Display for VectorRLWE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();
        to_be_print += " VectorRLWE {\n         -> samples = [";

        if self.ciphertexts.as_tensor().len() <= 2 * n {
            for elt in self.ciphertexts.as_tensor().iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        } else {
            for elt in self.ciphertexts.as_tensor().get_sub(0..n).iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
            to_be_print += "...";
            for elt in self
                .ciphertexts
                .as_tensor()
                .get_sub(self.ciphertexts.as_tensor().len() - n..)
                .iter()
            {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        }
        to_be_print += "]\n";
        to_be_print += "         -> variances = [";
        if self.variances.len() <= 2 * n {
            for elt in self.variances.iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
        } else {
            for elt in self.variances[0..n].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
            to_be_print += "...";
            // write!(f, "...");
            for elt in self.variances[self.variances.len() - n..].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
                // write!(f, "{}, ", elt);
            }
            // writeln!(f, "]");
        }
        to_be_print += "]\n";
        to_be_print = to_be_print + &format!("         -> dimension = {}\n", self.dimension);
        to_be_print =
            to_be_print + &format!("         -> polynomial_size = {}\n", self.polynomial_size);
        to_be_print =
            to_be_print + &format!("         -> nb of ciphertexts = {}\n", self.nb_ciphertexts);
        to_be_print += "       }\n";
        // writeln!(f, "         -> dimension = {}", self.dimension);
        // writeln!(f, "         -> polynomial_size = {}", self.polynomial_size);
        // writeln!(f, "         -> nb of ciphertexts = {}", self.nb_ciphertexts);
        // writeln!(f, "       }}")
        writeln!(f, "{}", to_be_print)
    }
}
