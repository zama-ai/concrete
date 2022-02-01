//! lwe ciphertext module

#[cfg(test)]
mod tests;

use crate::error::CryptoAPIError;
use crate::traits::GenericAdd;
use crate::{read_from_file, write_to_file, Torus};
use backtrace::Backtrace;
use colored::Colorize;
use concrete_commons::dispersion::StandardDev;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{GlweSize, LweSize, PolynomialSize};
use concrete_core::crypto::bootstrap::Bootstrap;
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;
use concrete_core::{
    crypto::{
        self,
        encoding::{Cleartext, Plaintext},
        glwe::GlweCiphertext,
        lwe::LweCiphertext,
    },
    math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor},
};
use concrete_npe as npe;
use concrete_npe::{LWE as NPELWE};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Structure containing a single LWE ciphertext.
///
/// # Attributes
/// * `ciphertext` - the LWE ciphertexts
/// * `variances` - the variance of the noise of the LWE ciphertext
/// * `dimension` - the length the LWE mask
/// * `encoder` - the encoder of the LWE ciphertext
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LWE {
    pub ciphertext: crypto::lwe::LweCiphertext<Vec<Torus>>,
    pub variance: f64,
    pub dimension: usize,
    pub encoder: crate::Encoder,
}

impl GenericAdd<f64, CryptoAPIError> for LWE {
    fn add(&self, right: f64) -> Result<LWE, CryptoAPIError> {
        self.add_constant_dynamic_encoder(right)
    }
    fn add_inplace(&mut self, right: f64) -> Result<(), CryptoAPIError> {
        self.add_constant_dynamic_encoder_inplace(right)
    }
}

impl GenericAdd<&LWE, CryptoAPIError> for LWE {
    fn add(&self, right: &LWE) -> Result<LWE, CryptoAPIError> {
        self.add_with_padding(right)
    }
    fn add_inplace(&mut self, right: &LWE) -> Result<(), CryptoAPIError> {
        self.add_with_padding_inplace(right)
    }
}

impl LWE {
    /// Instantiate a new LWE filled with zeros from a dimension
    ///
    /// # Arguments
    /// * `dimension` - the length the LWE mask
    ///
    /// # Output
    /// * a new instantiation of an LWE
    /// * ZeroCiphertextsInStructureError if we try to create a structure with no ciphertext in it
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // creates an LWE ciphertext with a dimension of 630
    /// let empty_ciphertexts = LWE::zero(630).unwrap();
    /// ```
    pub fn zero(dimension: usize) -> Result<crate::LWE, CryptoAPIError> {
        Ok(LWE {
            ciphertext: crypto::lwe::LweCiphertext::allocate(0, LweSize(dimension + 1)),
            variance: 0.,
            dimension,
            encoder: crate::Encoder::zero(),
        })
    }

    /// Encode a message and then directly encrypt the plaintext into an LWE structure
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `message` -  a  message as u64
    /// * `encoder` - an Encoder
    ///
    /// # Output
    /// an LWE structure
    ///
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(-2., 6., 4, 4).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a message
    /// let message: f64 = -1.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message, &encoder).unwrap();
    /// ```
    pub fn encode_encrypt(
        sk: &crate::LWESecretKey,
        message: f64,
        encoder: &crate::Encoder,
    ) -> Result<LWE, CryptoAPIError> {
        let plaintext = encoder.encode_core(message)?;
        let mut result_encoder: crate::Encoder = encoder.clone();
        let nb_bit_overlap: usize =
            result_encoder.update_precision_from_variance(f64::powi(sk.std_dev, 2i32))?;

        // notification of a problem
        if nb_bit_overlap > 0 {
            println!(
                "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension the reduce the amount of noise needed.",
                "Loss of precision during encrypt".red().bold(),
                nb_bit_overlap, encoder.nb_bit_precision
            );
        }

        let mut res = LWE {
            ciphertext: crypto::lwe::LweCiphertext::allocate(0, LweSize(sk.dimension + 1)),
            variance: 0.,
            dimension: sk.dimension,
            encoder: result_encoder,
        };
        res.encrypt_raw(sk, plaintext).unwrap();

        Ok(res)
    }

    /// Encrypt a raw plaintext (a Torus element instead of a struct Plaintext) with the provided key and standard deviation
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintext` - a Torus element
    /// * `std_dev` - the standard deviation used for the error normal distribution
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create one plaintext
    /// let pt: u64 = 0;
    ///
    /// // create one LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new LWE that encrypts pt
    /// let mut ct = LWE::zero(sk.dimension).unwrap();
    /// ct.encrypt_raw(&sk, pt).unwrap();
    /// ```
    pub fn encrypt_raw(
        &mut self,
        sk: &crate::LWESecretKey,
        plaintext: Torus,
    ) -> Result<(), CryptoAPIError> {
        // compute the variance
        let var = sk.get_variance();

        // check if we have enough std dev to have noise in the ciphertext
        if sk.std_dev < f64::powi(2., -(<Torus as Numeric>::BITS as i32) + 2) {
            return Err(NoNoiseInCiphertext!(var));
        }

        // fill the variance array
        self.variance = var;

        // encrypt
        sk.val.encrypt_lwe(
            &mut self.ciphertext,
            &Plaintext(plaintext),
            StandardDev::from_standard_dev(sk.std_dev),
            &mut EncryptionRandomGenerator::new(None),
        );

        Ok(())
    }

    /// Decrypt the ciphertext, meaning compute the phase and directly decode the output
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// # Output
    /// * `result` - a f64
    /// * DimensionError - if the ciphertext and the key have incompatible dimensions
    /// ```rust
    /// use concrete::*;
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list of messages in our interval
    /// let message: f64 = -3.2;
    ///
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new LWE that encrypts pt
    /// let mut ct = LWE::encode_encrypt(&sk,message, &encoder).unwrap();
    ///
    /// // decryption
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode(&self, sk: &crate::LWESecretKey) -> Result<f64, CryptoAPIError> {
        // check dimensions
        if sk.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, sk.dimension));
        }

        // create a temporary variable to store the result of the phase computation
        let mut output = Plaintext(0);

        // compute the phase
        sk.val.decrypt_lwe(&mut output, &self.ciphertext);

        // decode
        let result: f64 = self.encoder.decode_single(output.0)?;

        Ok(result)
    }

    /// Decrypt the ciphertext, meaning compute the phase and directly decode the output as if the encoder was in a rounding context
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// # Output
    /// * `result` - a f64
    /// * DimensionError - if the ciphertext and the key have incompatible dimensions
    /// ```rust
    /// use concrete::*;
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list of messages in our interval
    /// let message: f64 = -3.2;
    ///
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new LWE that encrypts pt
    /// let mut ct = LWE::encode_encrypt(&sk,message, &encoder).unwrap();
    ///
    /// // decryption
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode_round(&self, sk: &crate::LWESecretKey) -> Result<f64, CryptoAPIError> {
        // check dimensions
        if sk.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, sk.dimension));
        }

        // create a temporary variable to store the result of the phase computation
        let mut output = Plaintext(0);

        // compute the phase
        sk.val.decrypt_lwe(&mut output, &self.ciphertext);

        // round context
        let mut enc_round = self.encoder.clone();
        enc_round.round = true;

        // decode
        let result: f64 = enc_round.decode_single(output.0)?;

        Ok(result)
    }

    /// Add a small message to a LWE ciphertext and does not change the encoding but changes the bodies of the ciphertext
    ///
    /// # Argument
    /// * `message` - a f64
    ///
    /// # Output
    /// * a new LWE
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(100., 110., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 =-4.9;
    ///
    /// // encode and encrypt
    /// let ciphertext = LWE::encode_encrypt(&secret_key,message_1, &encoder).unwrap();
    ///
    /// // addition between ciphertext and message_2
    /// let ct_add = ciphertext.add_constant_static_encoder(message_2).unwrap();
    /// ```
    pub fn add_constant_static_encoder(&self, message: f64) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_static_encoder_inplace(message)?;
        Ok(res)
    }

    /// Add small messages to a LWE ciphertext and does not change the encoding but changes the bodies of the ciphertexts
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(100., 110., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 =-4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key,message_1, &encoder).unwrap();
    ///
    /// // addition between ciphertext and message_2
    /// ciphertext.add_constant_static_encoder_inplace(message_2).unwrap();
    /// ```
    pub fn add_constant_static_encoder_inplace(
        &mut self,
        message: f64,
    ) -> Result<(), CryptoAPIError> {
        // error if one message is not in [-delta,delta]
        if message.abs() > self.encoder.delta {
            return Err(MessageTooBigError!(message, self.encoder.delta));
        }
        let mut ec_tmp = self.encoder.clone();
        ec_tmp.o = 0.;
        let update = self
            .ciphertext
            .as_tensor()
            .get_element(self.dimension)
            .wrapping_add(ec_tmp.encode_outside_interval_operators(message)?);
        *self
            .ciphertext
            .as_mut_tensor()
            .get_element_mut(self.dimension) = update;

        Ok(())
    }

    /// Add a message to a LWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertext
    ///
    /// # Argument
    /// * `message` - a f64
    ///
    /// # Output
    /// * a new LWE
    /// * InvalidEncoderError if invalid encoder
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(100., 110., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = -4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    ///
    /// // addition between ciphertext and message_2
    /// let ct = ciphertext
    ///     .add_constant_dynamic_encoder(message_2)
    ///     .unwrap();
    /// ```
    pub fn add_constant_dynamic_encoder(&self, message: f64) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_dynamic_encoder_inplace(message)?;
        Ok(res)
    }

    /// Add a message to a LWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertext
    ///
    /// # Argument
    /// * `message` - a f64
    ///
    /// # Output
    /// * InvalidEncoderError if invalid encoder
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(100., 110., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64= 106.276;
    /// let message_2: f64 = -4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    ///
    /// // addition between ciphertext and message_2
    /// ciphertext
    ///     .add_constant_dynamic_encoder_inplace(message_2)
    ///     .unwrap();
    /// ```
    pub fn add_constant_dynamic_encoder_inplace(
        &mut self,
        message: f64,
    ) -> Result<(), CryptoAPIError> {
        if !self.encoder.is_valid() {
            return Err(InvalidEncoderError!(
                self.encoder.nb_bit_precision,
                self.encoder.delta
            ));
        }
        self.encoder.o += message;

        Ok(())
    }

    /// Compute an homomorphic addition between two LWE ciphertexts
    ///
    /// # Arguments
    /// * `ct` - an LWE struct
    /// * `new_min` - the min of the interval for the resulting Encoder
    ///
    /// # Output
    /// * a new LWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 0).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 =4.9;
    ///
    /// // new_min
    /// let new_min: f64 = 103.;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// let ct_add = ciphertext_1
    ///     .add_with_new_min(&ciphertext_2, new_min)
    ///     .unwrap();
    /// ```
    pub fn add_with_new_min(
        &self,
        ct: &crate::LWE,
        new_min: f64,
    ) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_new_min_inplace(ct, new_min)?;
        Ok(res)
    }

    /// Compute an homomorphic addition between two LWE ciphertexts
    ///
    /// # Arguments
    /// * `ct` - an LWE struct
    /// * `new_min` - the min of the interval for the resulting Encoder
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 0).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 0).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// // new_min
    /// let new_min: f64 = 103.;
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// ciphertext_1
    ///     .add_with_new_min_inplace(&ciphertext_2, new_min)
    ///     .unwrap();
    /// ```
    pub fn add_with_new_min_inplace(
        &mut self,
        ct: &crate::LWE,
        new_min: f64,
    ) -> Result<(), CryptoAPIError> {
        // check dimensions
        if ct.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // add the two ciphertexts together
        self.ciphertext.update_with_add(&ct.ciphertext);

        let enc1 = self.encoder.clone();
        let enc2 = ct.encoder.clone();

        // error if the deltas are not identical as well as the paddings
        if !deltas_eq!(enc1.delta, enc2.delta) {
            return Err(DeltaError!(enc1.delta, enc2.delta));
        } else if enc1.nb_bit_padding != enc2.nb_bit_padding {
            return Err(PaddingError!(enc1.nb_bit_padding, enc2.nb_bit_padding));
        }

        // correction related to the addition
        let mut tmp_ec = enc1.clone();
        tmp_ec.o = new_min;
        let tmp_sum = enc1.o + enc2.o;

        let lwe_body = self.ciphertext.get_body().0;
        let updated_body_value = if tmp_sum <= new_min {
            let tmp_shift = new_min + (new_min - tmp_sum);
            let plaintext = tmp_ec.encode_outside_interval_operators(tmp_shift)?;
            lwe_body.wrapping_sub(plaintext)
        } else {
            let plaintext = tmp_ec.encode_outside_interval_operators(tmp_sum)?;
            lwe_body.wrapping_add(plaintext)
        };
        self.ciphertext.get_mut_body().0 = updated_body_value;

        // update the Encoder
        self.encoder.o = new_min;

        // update the noise with the NPE
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }

    /// Compute an homomorphic addition between two LWE ciphertexts.
    /// The center of the output Encoder is the sum of the two centers of the input Encoders.
    /// # Arguments
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * a new LWE
    /// ```rust
    /// use concrete::*;
    ///
    /// let min_1: f64 = 85.;
    /// let min_2: f64 = -2.;
    /// let delta: f64 = 34.5;
    ///
    /// let max_1: f64 = min_1 + delta;
    /// let max_2: f64 = min_2 + delta;
    ///
    /// let (precision, padding) = (5, 2);
    /// let margin: f64 = 10.;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(min_1 - margin, max_1 + margin, precision, padding).unwrap();
    /// let encoder_2 = Encoder::new(min_2 - margin, max_2 + margin, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// let new_ciphertext = ciphertext_1.add_centered(&ciphertext_2).unwrap();
    /// ```
    pub fn add_centered(&self, ct: &crate::LWE) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_centered_inplace(ct)?;
        Ok(res)
    }

    /// Compute an homomorphic addition between two LWE ciphertexts.
    /// The center of the output Encoder is the sum of the two centers of the input Encoders
    ///
    /// # Arguments
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// let min_1: f64 = 85.;
    /// let min_2: f64 = -2.;
    /// let delta: f64 = 34.5;
    ///
    /// let max_1: f64 = min_1 + delta;
    /// let max_2: f64 = min_2 + delta;
    ///
    /// let (precision, padding) = (5, 2);
    /// let margin: f64 = 10.;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(min_1 - margin, max_1 + margin, precision, padding).unwrap();
    /// let encoder_2 = Encoder::new(min_2 - margin, max_2 + margin, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// ciphertext_1.add_centered_inplace(&ciphertext_2).unwrap();
    /// ```
    pub fn add_centered_inplace(&mut self, ct: &crate::LWE) -> Result<(), CryptoAPIError> {
        // check same dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // check same deltas

        if !deltas_eq!(self.encoder.delta, ct.encoder.delta) {
            return Err(DeltaError!(self.encoder.delta, ct.encoder.delta));
        }

        // add ciphertexts together
        self.ciphertext.update_with_add(&ct.ciphertext);

        // correction related to the addition
        let mut tmp_enc = self.encoder.clone();
        tmp_enc.o = 0.;
        let correction: Torus = tmp_enc.encode_core(self.encoder.delta / 2.)?;
        let update =
            self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_sub(correction);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // update the Encoder and the variance
        // compute the new encoder
        self.encoder.o += ct.encoder.o + self.encoder.delta / 2.;

        // compute the new variance
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);

        // update the encoder precision based on the variance
        self.encoder.update_precision_from_variance(self.variance)?;
        Ok(())
    }

    /// Compute an addition between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message stays the same: min(nb1,nb2)
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * a new LWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// let ct_add = ciphertext_1.add_with_padding(&ciphertext_2);
    /// ```
    pub fn add_with_padding(&self, ct: &crate::LWE) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an addition between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message stays the same: min(nb1,nb2)
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.add_with_padding_inplace(&ciphertext_2);
    /// ```
    pub fn add_with_padding_inplace(&mut self, ct: &crate::LWE) -> Result<(), CryptoAPIError> {
        // check same paddings
        if self.encoder.nb_bit_padding != ct.encoder.nb_bit_padding {
            return Err(PaddingError!(
                self.encoder.nb_bit_padding,
                ct.encoder.nb_bit_padding
            ));
        }
        // check at least one bit of padding
        else if self.encoder.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(self.encoder.nb_bit_padding, 1));
        }
        // check same deltas
        else if !deltas_eq!(self.encoder.delta, ct.encoder.delta) {
            return Err(DeltaError!(self.encoder.delta, ct.encoder.delta));
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // add ciphertexts together
        self.ciphertext.update_with_add(&ct.ciphertext);

        // update the Encoder list and variances

        // compute the new variance
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);

        // compute the new encoder
        self.encoder.o += ct.encoder.o;
        self.encoder.delta *= 2.;
        self.encoder.nb_bit_padding -= 1;
        self.encoder.nb_bit_precision =
            usize::min(self.encoder.nb_bit_precision, ct.encoder.nb_bit_precision);

        // update the encoder precision based on the variance
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }

    /// Compute an addition between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message increases: max(nb1,nb2) + 1
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * a new LWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(0., 255., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 255., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.;
    /// let message_2: f64 = 4.;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// let ct_add = ciphertext_1.add_with_padding_exact(&ciphertext_2);
    /// ```
    pub fn add_with_padding_exact(&self, ct: &crate::LWE) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_padding_exact_inplace(ct)?;
        Ok(res)
    }

    /// Compute an addition between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message increases: max(nb1,nb2) + 1
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(0., 255., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 255., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.;
    /// let message_2: f64 = 4.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.add_with_padding_exact_inplace(&ciphertext_2);
    /// ```
    pub fn add_with_padding_exact_inplace(
        &mut self,
        ct: &crate::LWE,
    ) -> Result<(), CryptoAPIError> {
        // check same paddings
        if self.encoder.nb_bit_padding != ct.encoder.nb_bit_padding {
            return Err(PaddingError!(
                self.encoder.nb_bit_padding,
                ct.encoder.nb_bit_padding
            ));
        }
        // check at least one bit of padding
        else if self.encoder.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(self.encoder.nb_bit_padding, 1));
        }
        // check same deltas
        else if !deltas_eq!(self.encoder.delta, ct.encoder.delta) {
            return Err(DeltaError!(self.encoder.delta, ct.encoder.delta));
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // add ciphertexts together
        self.ciphertext.update_with_add(&ct.ciphertext);

        // update the Encoder list and variances

        // compute the new variance
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);

        // compute the new encoder
        self.encoder.o += ct.encoder.o;
        self.encoder.delta *= 2.;
        self.encoder.nb_bit_padding -= 1;
        self.encoder.nb_bit_precision =
            usize::max(self.encoder.nb_bit_precision, ct.encoder.nb_bit_precision) + 1;

        // update the encoder precision based on the variance
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }

    /// Compute an subtraction between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message stays the same: min(nb1,nb2)
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * a new LWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// let ct_sub = ciphertext_1.add_with_padding(&ciphertext_2);
    /// ```
    pub fn sub_with_padding(&self, ct: &crate::LWE) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.sub_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an subtraction between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message stays the same: min(nb1,nb2)
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(100., 110., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 10., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.276;
    /// let message_2: f64 = 4.9;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.sub_with_padding_inplace(&ciphertext_2);
    /// ```
    pub fn sub_with_padding_inplace(&mut self, ct: &crate::LWE) -> Result<(), CryptoAPIError> {
        // check same paddings
        if self.encoder.nb_bit_padding != ct.encoder.nb_bit_padding {
            return Err(PaddingError!(
                self.encoder.nb_bit_padding,
                ct.encoder.nb_bit_padding
            ));
        }
        // check at least one bit of padding
        else if self.encoder.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(self.encoder.nb_bit_padding, 1));
        }
        // check same deltas
        else if !deltas_eq!(self.encoder.delta, ct.encoder.delta) {
            return Err(DeltaError!(self.encoder.delta, ct.encoder.delta));
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // subtract ciphertexts together
        self.ciphertext.update_with_sub(&ct.ciphertext);

        // correction related to the subtraction
        let correction: Torus = 1 << (<Torus as Numeric>::BITS - self.encoder.nb_bit_padding);
        let update =
            self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_add(correction);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // update the Encoder
        self.encoder.o -= ct.encoder.o + ct.encoder.delta;
        self.encoder.delta *= 2.;
        self.encoder.nb_bit_padding -= 1;
        self.encoder.nb_bit_precision =
            usize::min(self.encoder.nb_bit_precision, ct.encoder.nb_bit_precision);

        // update the noise with the NPE
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }

    /// Compute an subtraction between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message increases: max(nb1,nb2) + 1
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * a new LWE
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(0., 255., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 255., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.;
    /// let message_2: f64 = 4.;
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// let ct_sub = ciphertext_1.sub_with_padding_exact(&ciphertext_2);
    /// ```
    pub fn sub_with_padding_exact(&self, ct: &crate::LWE) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.sub_with_padding_exact_inplace(ct)?;
        Ok(res)
    }

    /// Compute an subtraction between two LWE ciphertexts by eating one bit of padding.
    /// Note that the number of bits of message increases: max(nb1,nb2) + 1
    ///
    /// # Argument
    /// * `ct` - an LWE struct
    ///
    /// # Output
    /// * DimensionError - if the ciphertexts have incompatible dimensions
    /// * DeltaError - if the ciphertexts have incompatible deltas
    /// * PaddingError - if the ciphertexts have incompatible paddings
    /// * NotEnoughPaddingError - if nb bit of padding is zero
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(0., 255., 8, 1).unwrap();
    /// let encoder_2 = Encoder::new(0., 255., 8, 1).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 106.;
    /// let message_2: f64 = 4.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
    /// let ciphertext_2 = LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.sub_with_padding_exact_inplace(&ciphertext_2);
    /// ```
    pub fn sub_with_padding_exact_inplace(
        &mut self,
        ct: &crate::LWE,
    ) -> Result<(), CryptoAPIError> {
        // check same paddings
        if self.encoder.nb_bit_padding != ct.encoder.nb_bit_padding {
            return Err(PaddingError!(
                self.encoder.nb_bit_padding,
                ct.encoder.nb_bit_padding
            ));
        }
        // check at least one bit of padding
        else if self.encoder.nb_bit_padding == 0 {
            return Err(NotEnoughPaddingError!(self.encoder.nb_bit_padding, 1));
        }
        // check same deltas
        else if !deltas_eq!(self.encoder.delta, ct.encoder.delta) {
            return Err(DeltaError!(self.encoder.delta, ct.encoder.delta));
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // subtract ciphertexts together
        self.ciphertext.update_with_sub(&ct.ciphertext);

        // correction related to the subtraction
        let correction: Torus = 1 << (<Torus as Numeric>::BITS - self.encoder.nb_bit_padding);
        let update =
            self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_add(correction);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // update the Encoder
        self.encoder.o -= ct.encoder.o + ct.encoder.delta;
        self.encoder.delta *= 2.;
        self.encoder.nb_bit_padding -= 1;
        self.encoder.nb_bit_precision =
            usize::max(self.encoder.nb_bit_precision, ct.encoder.nb_bit_precision) + 1;

        // update the noise with the NPE
        self.variance = npe::add_ciphertexts(self.variance, ct.variance);
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }

    /// Multiply LWE ciphertext with small integer message and does not change the encoding but changes the body and mask of the ciphertext
    ///
    /// # Argument
    /// * `messages` - a list of integer messages
    ///
    /// # Output
    /// * a new LWE
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let b = min.abs().min(max.abs()) / 20.;
    /// let precision = 6;
    /// let padding = 2;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 6.923;
    /// let message_2: i32= 2;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    /// let new_ciphertext = ciphertext.mul_constant_static_encoder(message_2).unwrap();
    /// ```
    pub fn mul_constant_static_encoder(&self, message: i32) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.mul_constant_static_encoder_inplace(message)?;
        Ok(res)
    }

    /// Multiply LWE ciphertext with small integer message and does not change the encoding but changes the body and mask of the ciphertext
    ///
    /// # Argument
    /// * `messages` - a list of integer messages
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let b = min.abs().min(max.abs()) / 20.;
    /// let precision = 6;
    /// let padding = 2;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = 6.923;
    /// let message_2: i32= 2;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    /// ciphertext
    ///     .mul_constant_static_encoder_inplace(message_2)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_static_encoder_inplace(
        &mut self,
        message: i32,
    ) -> Result<(), CryptoAPIError> {
        // compute correction
        let cor0: Torus = self.encoder.encode_outside_interval_operators(0.)?;
        let cor = cor0.wrapping_mul((message - 1) as Torus);

        // multiplication
        self.ciphertext
            .update_with_scalar_mul(Cleartext(message as Torus));

        // apply correction
        let update = self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_sub(cor);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // compute the absolute value
        let m_abs = message.abs();

        // call to the NPE to estimate the new variance
        self.variance = npe::LWE::single_scalar_mul(self.variance, m_abs as Torus);

        if m_abs != 0 {
            // update the encoder precision based on the variance
            self.encoder.update_precision_from_variance(self.variance)?;
        }

        Ok(())
    }

    /// Multiply each LWE ciphertext with a real constant and do change the encoding and the ciphertexts by consuming some bits of padding
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
    /// * a new LWE
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 6;
    /// let padding = precision + 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = -106.276;
    /// let message_2: f64 = 2.432;
    /// let b: f64 = 6.;
    ///
    /// // encode and encrypt
    /// let ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    /// let new_ciphertext = ciphertext
    ///     .mul_constant_with_padding(message_2, b, precision)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_with_padding(
        &self,
        constant: f64,
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.mul_constant_with_padding_inplace(constant, max_constant, nb_bit_padding)?;
        Ok(res)
    }

    /// Multiply each LWE ciphertext with a real constant and do change the encoding and the ciphertexts by consuming some bits of padding
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
    /// * ConstantMaximumError - if one element of `constants` if bigger than `max_constant`
    /// * ZeroInIntervalError - if zero is not in the interval described by the encoders
    /// * NotEnoughPaddingError - if there is not enough padding
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 6;
    /// let padding = precision + 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = -106.276;
    /// let message_2: f64 = 2.432;
    /// let b: f64 = 6.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    /// ciphertext
    ///     .mul_constant_with_padding_inplace(message_2, b, precision)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_with_padding_inplace(
        &mut self,
        constant: f64,
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<(), CryptoAPIError> {
        // check that the constant if below the maximum
        if constant > max_constant || constant < -max_constant {
            return Err(ConstantMaximumError!(constant, max_constant));
        }
        // check that zero is in the interval
        else if self.encoder.o > 0. || self.encoder.o + self.encoder.delta < 0. {
            return Err(ZeroInIntervalError!(self.encoder.o, self.encoder.delta));
        }
        // check bits of paddings
        else if self.encoder.nb_bit_padding < nb_bit_padding {
            return Err(NotEnoughPaddingError!(
                self.encoder.nb_bit_padding,
                nb_bit_padding
            ));
        }

        // test if negative
        let negative: bool = constant < 0.;

        // absolute value
        let c_abs = constant.abs();

        // discretize c_abs with regard to the number of bits of padding to use
        let scal: Torus =
            (c_abs / max_constant * f64::powi(2., nb_bit_padding as i32)).round() as Torus;

        // encode 0 and subtract it
        let tmp_sub = self.encoder.encode_core(0.)?;
        let update = self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_sub(tmp_sub);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // scalar multiplication
        self.ciphertext.update_with_scalar_mul(Cleartext(scal));

        // new encoder
        let new_o = self.encoder.o * max_constant;
        let new_max =
            (self.encoder.o + self.encoder.delta - self.encoder.get_granularity()) * max_constant;
        let new_delta = new_max - new_o;

        // compute the discretization of c_abs
        let discret_c_abs = (scal as f64) * f64::powi(2., -(nb_bit_padding as i32)) * max_constant;

        // compute  the rounding error on c_abs
        let rounding_error = (discret_c_abs - c_abs).abs();

        // get the ciphertext granularity
        let granularity = self.encoder.get_granularity();

        // compute the max of the ciphertext (based on the metadata of the encoder)
        let max = f64::max(
            (self.encoder.o + self.encoder.delta - self.encoder.get_granularity()).abs(),
            self.encoder.o.abs(),
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
            self.encoder.nb_bit_precision,
        );

        // create the new encoder
        let tmp_encoder = crate::Encoder::new(
            new_o,
            new_max,
            usize::min(nb_bit_padding, self.encoder.nb_bit_precision),
            self.encoder.nb_bit_padding - nb_bit_padding,
        )?;
        self.encoder.copy(&tmp_encoder);
        self.encoder.nb_bit_precision = usize::min(self.encoder.nb_bit_precision, new_precision);
        // call to the NPE to estimate the new variance
        self.variance = npe::LWE::single_scalar_mul(self.variance, scal);

        if scal != 0 {
            // update the encoder precision based on the variance
            self.encoder.update_precision_from_variance(self.variance)?;
        }

        // encode 0 with the new encoder
        let tmp_add = self.encoder.encode_core(0.)?;
        let update = self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_add(tmp_add);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        if negative {
            // compute the opposite
            self.ciphertext.update_with_neg();

            // add correction if there is some padding
            let mut cor: Torus = 0;
            if self.encoder.nb_bit_padding > 0 {
                cor = (1 << (<Torus as Numeric>::BITS - self.encoder.nb_bit_padding))
                    - (1 << (<Torus as Numeric>::BITS
                        - self.encoder.nb_bit_padding
                        - self.encoder.nb_bit_precision));
            } else {
                cor = cor.wrapping_sub(
                    1 << (<Torus as Numeric>::BITS
                        - self.encoder.nb_bit_padding
                        - self.encoder.nb_bit_precision),
                );
            }
            let update = self
                .ciphertext
                .as_tensor()
                .get_element(self.dimension)
                .wrapping_add(cor);
            *self
                .ciphertext
                .as_mut_tensor()
                .get_element_mut(self.dimension) = update;

            // change the encoder
            self.encoder.opposite_inplace()?;
        }

        Ok(())
    }

    /// Compute the opposite of the n-th LWE ciphertext in the structure
    ///
    /// # Output
    /// * a new LWE ciphertext
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 6;
    /// let padding = 5;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 =-106.276;
    ///
    /// // encode and encrypt
    /// let ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    ///
    /// let new_ciphertext = ciphertext.opposite().unwrap();
    /// ```
    pub fn opposite(&self) -> Result<crate::LWE, CryptoAPIError> {
        let mut res = self.clone();
        res.opposite_inplace()?;
        Ok(res)
    }

    /// Compute the opposite of the n-th LWE ciphertext in the structure
    ///
    /// # Output
    /// * IndexError - if the requested ciphertext does not exist
    /// * InvalidEncoderError - if the encoder of the requested ciphertext is not valid (i.e. with nb_bit_precision = 0 or delta = 0)
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 6;
    /// let padding = 5;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message_1: f64 = -106.276;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
    ///
    /// ciphertext.opposite_inplace().unwrap();
    /// ```
    pub fn opposite_inplace(&mut self) -> Result<(), CryptoAPIError> {
        // check the encoders
        if !self.encoder.is_valid() {
            return Err(InvalidEncoderError!(
                self.encoder.nb_bit_precision,
                self.encoder.delta
            ));
        }

        // compute the opposite
        self.ciphertext.update_with_neg();

        // add correction if there is some padding
        let mut cor: Torus = 0;
        if self.encoder.nb_bit_padding > 0 {
            cor = (1 << (<Torus as Numeric>::BITS - self.encoder.nb_bit_padding))
                - (1 << (<Torus as Numeric>::BITS
                    - self.encoder.nb_bit_padding
                    - self.encoder.nb_bit_precision));
        } else {
            cor = cor.wrapping_sub(
                1 << (<Torus as Numeric>::BITS
                    - self.encoder.nb_bit_padding
                    - self.encoder.nb_bit_precision),
            );
        }
        let update = self.ciphertext.as_tensor().as_slice()[self.dimension].wrapping_add(cor);
        self.ciphertext.as_mut_tensor().as_mut_slice()[self.dimension] = update;

        // change the encoder
        self.encoder.opposite_inplace()?;

        Ok(())
    }

    /// Compute a key switching operation on every ciphertext from the LWE struct self
    ///
    /// # Argument
    /// * `ksk` - the key switching key
    ///
    /// # Output
    /// * a LWE struct
    ///
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 6;
    /// let padding = 1;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let message: f64 = -106.276;
    ///
    /// // generate two secret keys
    /// let secret_key_before = LWESecretKey::new(&LWE128_1024);
    /// let secret_key_after = LWESecretKey::new(&LWE128_1024);
    ///
    /// // generate the key switching key
    /// let ksk = LWEKSK::new(&secret_key_before, &secret_key_after, base_log, level);
    ///
    /// // a list of messages that we encrypt
    /// let ciphertext_before =
    ///     LWE::encode_encrypt(&secret_key_before, message, &encoder).unwrap();
    ///
    /// // key switch
    /// let ciphertext_after = ciphertext_before.keyswitch(&ksk).unwrap();
    /// ```
    pub fn keyswitch(&self, ksk: &crate::LWEKSK) -> Result<crate::LWE, CryptoAPIError> {
        // allocation for the result
        let mut res: crate::LWE = crate::LWE::zero(ksk.dimension_after)?;

        // key switch
        ksk.ciphertexts
            .keyswitch_ciphertext(&mut res.ciphertext, &self.ciphertext);

        // deal with encoders, noise and new precision
        // calls the NPE to find out the amount of noise after KS
        res.variance = <Torus as NPELWE>::key_switch(
            self.dimension,
            ksk.level,
            ksk.base_log,
            ksk.variance,
            self.variance,
        );

        // copy the encoders
        res.encoder.copy(&self.encoder);

        // update the precision
        let nb_bit_overlap: usize = res.encoder.update_precision_from_variance(res.variance)?;

        // notification of a problem
        if nb_bit_overlap > 0 {
            println!(
                "{}: {} bit(s) lost, with {} bit(s) of message originally",
                "Loss of precision during key switch".red().bold(),
                nb_bit_overlap,
                self.encoder.nb_bit_precision
            );
        }

        Ok(res)
    }

    /// Compute a bootstrap on the LWE
    ///
    /// # Argument
    /// * `bsk` - the bootstrapping key
    ///
    /// # Output
    /// * a LWE struct
    /// * IndexError - if the requested ciphertext does not exist
    /// * DimensionError - if the bootstrapping key and the input ciphertext have incompatible dimensions
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 4;
    /// let padding = 1;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a message
    /// let message: f64 = -106.276;
    ///
    /// // generate two secret keys
    /// let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
    /// let secret_key_before = LWESecretKey::new(&LWE128_630);
    /// let secret_key_after = rlwe_secret_key.to_lwe_secret_key();
    ///
    /// // bootstrapping key
    /// let bootstrapping_key =
    ///     LWEBSK::new(&secret_key_before, &rlwe_secret_key, base_log, level);
    ///
    /// // encode and encrypt
    /// let ciphertext_before =
    ///     LWE::encode_encrypt(&secret_key_before,message, &encoder).unwrap();
    ///
    /// let ciphertext_out = ciphertext_before
    ///     .bootstrap(&bootstrapping_key)
    ///     .unwrap();
    /// ```
    pub fn bootstrap(&self, bsk: &crate::LWEBSK) -> Result<crate::LWE, CryptoAPIError> {
        self.bootstrap_with_function(bsk, |x| x, &self.encoder)
    }

    /// Compute a bootstrap and apply an arbitrary function to the LWE ciphertext
    ///
    /// # Argument
    /// * `bsk` - the bootstrapping key
    /// * `f` - the function to aply
    /// * `encoder_output` - a list of output encoders
    ///
    /// # Output
    /// * a LWE struct
    /// * IndexError - if the requested ciphertext does not exist
    /// * DimensionError - if the bootstrapping key and the input ciphertext have incompatible dimensions
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 4;
    /// let padding = 1;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder_input = Encoder::new(min, max, precision, padding).unwrap();
    /// let encoder_output = Encoder::new(0., max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a message
    /// let message: f64 = -106.276;
    ///
    /// // generate secret keys
    /// let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
    /// let secret_key_before = LWESecretKey::new(&LWE128_630);
    /// let secret_key_after = rlwe_secret_key.to_lwe_secret_key();
    ///
    /// // bootstrapping key
    /// let bootstrapping_key =
    ///     LWEBSK::new(&secret_key_before, &rlwe_secret_key, base_log, level);
    ///
    /// // encode and encrypt
    /// let ciphertext_before =
    ///     LWE::encode_encrypt(&secret_key_before, message, &encoder_input).unwrap();
    ///
    /// let ciphertext_out = ciphertext_before
    ///     .bootstrap_with_function(&bootstrapping_key, |x| f64::max(0., x), &encoder_output)
    ///     .unwrap();
    /// ```
    pub fn bootstrap_with_function<F: Fn(f64) -> f64>(
        &self,
        bsk: &crate::LWEBSK,
        f: F,
        encoder_output: &crate::Encoder,
    ) -> Result<crate::LWE, CryptoAPIError> {
        // check bsk compatibility
        if self.dimension != bsk.get_lwe_dimension() {
            return Err(DimensionError!(self.dimension, bsk.get_lwe_dimension()));
        }

        // generate the look up table (throw error if a bit of padding is missing)
        let lut = bsk.generate_functional_look_up_table(&self.encoder, encoder_output, f)?;

        // build the trivial accumulator
        let mut accumulator = GlweCiphertext::allocate(
            0_u64,
            PolynomialSize(bsk.polynomial_size),
            GlweSize(bsk.dimension + 1),
        );
        accumulator
            .as_mut_tensor()
            .as_mut_slice()
            .get_mut(
                (bsk.dimension * bsk.polynomial_size)..((bsk.dimension + 1) * bsk.polynomial_size),
            )
            .unwrap()
            .copy_from_slice(&lut);

        // allocate the result
        let mut result =
            LweCiphertext::allocate(0, LweSize(bsk.dimension * bsk.polynomial_size + 1));

        if self.encoder.nb_bit_padding > 1 {
            // remove the padding but one bit
            let mut self_clone = self.clone();
            self_clone.remove_padding_inplace(self.encoder.nb_bit_padding - 1)?;

            // compute the bootstrap
            bsk.ciphertexts
                .bootstrap(&mut result, &self_clone.ciphertext, & accumulator);
        } else {
            // compute the bootstrap
            bsk.ciphertexts
                .bootstrap(&mut result, &self.ciphertext, & accumulator);
        }

        // compute the new variance (without the drift)
        let new_var: f64 = <Torus as npe::cross::Cross>::bootstrap(
            self.dimension,
            bsk.dimension,
            bsk.level,
            bsk.base_log,
            bsk.polynomial_size,
            bsk.variance,
        );

        // create the output encoder
        let mut new_encoder_output: crate::Encoder = encoder_output.clone();

        // update the precision in case of the output noise (without drift) is too big and overlap the message
        let nb_bit_overlap: usize = new_encoder_output.update_precision_from_variance(new_var)?;

        // println!("call to npe : {}", new_var);
        if nb_bit_overlap > 0 {
            println!(
                "{}: {} bit(s) of precision lost over {} bit(s) of message originally. Consider increasing the number of level and/or decreasing the log base.",
                "Loss of precision during bootstrap".red().bold(),
                nb_bit_overlap, self.encoder.nb_bit_precision
            );
        }

        // calls the NPE to find out the amount of noise after rounding the input ciphertext (drift)
        let nb_rounding_noise_bit: usize =
            (npe::lwe::log2_rounding_noise(self.dimension)).ceil() as usize + 1;

        // deals with the drift error
        if nb_rounding_noise_bit + 1 + new_encoder_output.nb_bit_precision
            > bsk.get_polynomial_size_log() + 1
        {
            let nb_bit_loss = 1 + new_encoder_output.nb_bit_precision + nb_rounding_noise_bit
                - bsk.get_polynomial_size_log()
                - 1;

            new_encoder_output.nb_bit_precision = i32::max(
                new_encoder_output.nb_bit_precision as i32 - nb_bit_loss as i32,
                0i32,
            ) as usize;
            // drift
            println!(
                "{}: {} bit(s) of precision lost over {} bit(s) of message originally ({} bits are affected by the noise). Consider increasing the polynomial size of the RLWE secret key.",
                "Loss of precision during bootstrap due to the rounding".red().bold(),
                nb_bit_loss, self.encoder.nb_bit_precision,nb_rounding_noise_bit
            );
        }

        // construct the output
        let lwe = crate::LWE {
            variance: new_var,
            ciphertext: result,
            dimension: bsk.polynomial_size * bsk.dimension,
            encoder: new_encoder_output,
        };

        Ok(lwe)
    }

    /// Multiply two LWE ciphertexts thanks to two bootstrapping procedures
    /// need to have 2 bits of padding at least
    ///
    /// # Argument
    /// * `ct` - an LWE struct containing the second LWE for the multiplication
    /// * `bsk` - the bootstrapping key used to evaluate the function
    /// * `n_self` - the index of the ciphertext to multiply in the self struct
    /// * `n_ct` - the index of the ciphertext to multiply in the ct struct
    ///
    /// # Output
    /// * a LWE struct
    /// * NotEnoughPaddingError - if one of the input does not have at least 2 bits of padding
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min_1, max_1): (f64, f64) = (-150., 204.);
    /// let min_2: f64 = 30.;
    /// let max_2: f64 = min_2 + max_1 - min_1;
    ///
    /// let precision = 4;
    /// let padding = 2;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder_1 = Encoder::new(min_1, max_1, precision, padding).unwrap();
    /// let encoder_2 = Encoder::new(min_2, max_2, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_650);
    ///
    /// // two lists of messages
    /// let message_1: f64 = -127.;
    /// let message_2: f64 = 72.7;
    ///
    /// // generate secret keys
    /// let rlwe_secret_key = RLWESecretKey::new(&RLWE128_1024_1);
    /// let secret_key_before = LWESecretKey::new(&LWE128_630);
    /// let secret_key_after = rlwe_secret_key.to_lwe_secret_key();
    ///
    /// // bootstrapping key
    /// let bootstrapping_key =
    ///     LWEBSK::new(&secret_key_before, &rlwe_secret_key, base_log, level);
    ///
    /// // a list of messages that we encrypt
    /// let ciphertext_1 =
    ///     LWE::encode_encrypt(&secret_key_before, message_1, &encoder_1).unwrap();
    ///
    /// let ciphertext_2 =
    ///     LWE::encode_encrypt(&secret_key_before, message_2, &encoder_2).unwrap();
    ///
    /// let ciphertext_out = ciphertext_1
    ///     .mul_from_bootstrap(&ciphertext_2, &bootstrapping_key)
    ///     .unwrap();
    /// ```
    pub fn mul_from_bootstrap(
        &self,
        ct: &crate::LWE,
        bsk: &crate::LWEBSK,
    ) -> Result<crate::LWE, CryptoAPIError> {
        // clone twice from self
        let mut ct1 = self.clone();
        let mut ct2 = self.clone();

        // return an error if nb_bit_precision < 2
        if ct1.encoder.nb_bit_precision < 2 {
            return Err(NotEnoughPaddingError!(ct1.encoder.nb_bit_precision, 2));
        }

        // extract once from ct
        let input2 = ct.clone();

        // compute the addition and the subtraction
        ct1.add_with_padding_inplace(&input2)?;
        ct2.sub_with_padding_inplace(&input2)?;

        // create the output encoder
        let mut encoder_out1 = ct1.encoder.new_square_divided_by_four(2)?;
        let mut encoder_out2 = ct2.encoder.new_square_divided_by_four(2)?;

        // set the same deltas
        if encoder_out1.delta < encoder_out2.delta {
            encoder_out1.delta = encoder_out2.delta;
        } else {
            encoder_out2.delta = encoder_out1.delta;
        }

        // bootstrap
        let mut square1 = ct1.bootstrap_with_function(bsk, |x| (x * x) / 4., &encoder_out1)?;
        let square2 = ct2.bootstrap_with_function(bsk, |x| (x * x) / 4., &encoder_out2)?;

        // subtract
        square1.sub_with_padding_inplace(&square2)?;

        Ok(square1)
    }

    /// Return the size of one LWE ciphertext with the parameters of self
    ///
    /// # Output
    /// * a usize with the size of a single LWE ciphertext
    pub fn get_ciphertext_size(&self) -> usize {
        self.dimension + 1
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        write_to_file(path, self)
    }

    pub fn load(path: &str) -> Result<LWE, Box<dyn Error>> {
        read_from_file(path)
    }

    /// Removes nb bits of padding
    ///
    /// # Arguments
    /// * `nb` - number of bits of padding to remove
    ///
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(-2., 6., 4, 4).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a message
    /// let message: f64 = -1.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = LWE::encode_encrypt(&secret_key, message, &encoder).unwrap();
    ///
    /// // removing 2 bits of padding
    /// ciphertext.remove_padding_inplace(2).unwrap();
    /// ```
    pub fn remove_padding_inplace(&mut self, nb: usize) -> Result<(), CryptoAPIError> {
        // check that te input encoder has at least 1 bit of padding
        if self.encoder.nb_bit_padding < nb {
            return Err(NotEnoughPaddingError!(self.encoder.nb_bit_padding, nb));
        }

        // shift of nb bits to the left
        self.ciphertext.as_mut_tensor().update_with_scalar_shl(&nb);

        // correction of the encoder
        self.encoder.nb_bit_padding -= nb;

        // call to the NPE to estimate the new variance
        let coeff: Torus = 1 << nb;
        self.variance = npe::LWE::single_scalar_mul(self.variance, coeff);

        // update the encoder precision based on the variance
        self.encoder.update_precision_from_variance(self.variance)?;

        Ok(())
    }
}

/// Print needed pieces of information about an LWE
impl fmt::Display for LWE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " LWE {\n         -> samples = [";

        if self.ciphertext.as_tensor().len() <= 2 * n {
            for elt in self.ciphertext.as_tensor().iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        } else {
            for elt in self.ciphertext.as_tensor().get_sub(0..n).iter() {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
            to_be_print += "...";

            for elt in self
                .ciphertext
                .as_tensor()
                .get_sub(self.ciphertext.as_tensor().len() - n..)
                .iter()
            {
                to_be_print = to_be_print + &format!("{}, ", *elt);
            }
        }
        to_be_print += "]\n";

        to_be_print += &format!("         -> variance = {}\n", self.variance);
        to_be_print = to_be_print + &format!("         -> dimension = {}\n", self.dimension);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
