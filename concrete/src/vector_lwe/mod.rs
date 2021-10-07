//! vector_lwe ciphertext module

use std::error::Error;
use std::fmt;

use crate::error::CryptoAPIError;
use crate::{read_from_file, write_to_file, Torus};
use backtrace::Backtrace;
use colored::Colorize;
use concrete_commons::dispersion::StandardDev;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{CiphertextCount, GlweSize, LweSize, PolynomialSize};
use concrete_core::crypto::secret::generators::EncryptionRandomGenerator;
use concrete_core::{
    crypto::{
        encoding::PlaintextList,
        glwe::GlweCiphertext,
        lwe::{LweCiphertext, LweList},
    },
    math::tensor::Tensor,
    math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor},
};

use concrete_core::crypto::bootstrap::Bootstrap;

use concrete_npe as npe;
use concrete_npe::{ LWE};
use itertools::izip;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

/// Structure containing a list of LWE ciphertexts.
/// They all have the same dimension (i.e. the length of the LWE mask).
///
/// # Attributes
/// * `ciphertexts` - the concatenation of all the LWE ciphertexts of the list
/// * `variances` - the variances of the noise of each LWE ciphertext of the list
/// * `dimension` - the length the LWE mask
/// * `nb_ciphertexts` - the number of LWE ciphertexts present in the list
/// * `encoders` - the encoders of each LWE ciphertext of the list
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorLWE {
    pub ciphertexts: LweList<Vec<Torus>>,
    pub variances: Vec<f64>,
    pub dimension: usize,
    pub nb_ciphertexts: usize,
    pub encoders: Vec<crate::Encoder>,
}

impl VectorLWE {
    /// Instantiate a new VectorLWE filled with zeros from a dimension and a number of ciphertexts
    /// `nb_ciphertexts` has to be at least 1.
    ///
    /// # Arguments
    /// * `dimension` - the length the LWE mask
    /// * `nb_ciphertexts` - the number of LWE ciphertexts to be stored in the structure
    ///
    /// # Output
    /// * a new instantiation of an VectorLWE
    /// * ZeroCiphertextsInStructureError if we try to create a structure with no ciphertext in it
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // creates a list of 5 empty LWE ciphertexts with a dimension of 630
    /// let empty_ciphertexts = VectorLWE::zero(630, 5).unwrap();
    /// ```
    pub fn zero(
        dimension: usize,
        nb_ciphertexts: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        if nb_ciphertexts == 0 {
            return Err(ZeroCiphertextsInStructureError!(nb_ciphertexts));
        }
        Ok(VectorLWE {
            ciphertexts: LweList::allocate(
                0,
                LweSize(dimension + 1),
                CiphertextCount(nb_ciphertexts),
            ),
            variances: vec![0.; nb_ciphertexts],
            dimension,
            nb_ciphertexts,
            encoders: vec![crate::Encoder::zero(); nb_ciphertexts],
        })
    }

    /// Copy one ciphertext from an VectorLWE structure inside the self VectorLWE structure
    /// i.e. copy the ct_index-th LWE ciphertext from ct inside the self_index-th of self
    ///
    /// # Arguments
    /// * `self_index` - the index in self we will paste the ciphertext
    /// * `ct` - the VectorLWE structure we will copy a ciphertext from
    /// * `ct_index` - the index of the ciphertext in ct we will copy
    ///
    /// # Output
    /// * DimensionError if self and ct does not share the same dimension
    /// * IndexError if self_index >= self.nb_ciphertexts
    /// * IndexError if ct_index >= ct.nb_ciphertexts
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    /// // creates a list of 5 empty LWE ciphertexts with a dimension of 630
    /// let mut ct1 = VectorLWE::zero(630, 5).unwrap();
    /// // creates a list of 8 empty LWE ciphertexts with a dimension of 630
    /// let ct2 = VectorLWE::zero(630, 8).unwrap();
    ///
    /// // copy the last ciphertext of ct2 at the first position of ct1
    /// ct1.copy_in_nth_nth_inplace(0, &ct2, 7).unwrap();
    /// ```
    pub fn copy_in_nth_nth_inplace(
        &mut self,
        self_index: usize,
        ct: &VectorLWE,
        ct_index: usize,
    ) -> Result<(), CryptoAPIError> {
        // check dimensions
        if ct.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // check self_index
        if self_index >= self.nb_ciphertexts {
            return Err(IndexError!(self.nb_ciphertexts, self_index));
        }

        // check ct_index
        if ct_index >= ct.nb_ciphertexts {
            return Err(IndexError!(ct.nb_ciphertexts, ct_index));
        }

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        // copy the content
        for (output, input) in izip!(
            self.ciphertexts
                .as_mut_tensor()
                .get_sub_mut((self_index * (ct_size))..((self_index + 1) * (ct_size)))
                .iter_mut(),
            ct.ciphertexts
                .as_tensor()
                .get_sub((ct_index * (ct_size))..((ct_index + 1) * (ct_size)))
                .iter()
        ) {
            *output = *input;
        }

        // copy the variance
        self.variances[self_index] = ct.variances[ct_index];

        // copy the encoder
        self.encoders[self_index].copy(&ct.encoders[ct_index]);

        Ok(())
    }

    /// extract the n-th of the LWE ciphertexts from an VectorLWE structure and output a new VectorLWE structure with only a copy of this ciphertext
    ///
    /// # Arguments
    /// * `n` - the index the ciphertext to extract
    ///
    /// # Output
    /// * IndexError if n >= self.nb_ciphertexts
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // creates a list of 6 empty LWE ciphertexts with a dimension of 630
    /// let ct = VectorLWE::zero(630, 6).unwrap();
    ///
    /// // extract the first ciphertext of ct
    /// let ct_extracted = ct.extract_nth(0).unwrap();
    /// ```
    pub fn extract_nth(&self, n: usize) -> Result<crate::VectorLWE, CryptoAPIError> {
        if n >= self.nb_ciphertexts {
            return Err(IndexError!(self.nb_ciphertexts, n));
        }

        let ct = self
            .ciphertexts
            .as_tensor()
            .get_sub((n * (self.get_ciphertext_size()))..((n + 1) * (self.get_ciphertext_size())))
            .as_slice()
            .to_owned();
        let result = VectorLWE {
            ciphertexts: LweList::from_container(ct.to_vec(), LweSize(self.dimension + 1)),
            variances: vec![self.variances[n]; 1],
            dimension: self.dimension,
            nb_ciphertexts: 1,
            encoders: vec![self.encoders[n].clone(); 1],
        };
        Ok(result)
    }

    /// Encrypt plaintexts from a Plaintext with the provided LWEParams
    ///
    /// # Arguments
    /// * `sk` - an LWESecretKey
    /// * `plaintexts` - a Plaintext
    ///
    /// # Output
    /// * VectorLWE structure
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
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
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let ct = VectorLWE::encrypt(&sk, &pt);
    /// ```
    pub fn encrypt(
        sk: &crate::LWESecretKey,
        plaintexts: &crate::Plaintext,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = VectorLWE::zero(sk.dimension, plaintexts.nb_plaintexts)?;
        res.encrypt_inplace(sk, plaintexts)?;
        Ok(res)
    }

    /// Encode messages and then directly encrypt the plaintexts into an VectorLWE structure
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `messages` -  a list of messages as u64
    /// * `encoder` - an Encoder
    ///
    /// # Output
    /// an VectorLWE structure
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // encoder
    /// let encoder = Encoder::new(-2., 6., 4, 4).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let messages: Vec<f64> = vec![-1., 2., 0., 5., -0.5];
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
    /// ```
    pub fn encode_encrypt(
        sk: &crate::LWESecretKey,
        messages: &[f64],
        encoder: &crate::Encoder,
    ) -> Result<VectorLWE, CryptoAPIError> {
        let mut plaintexts: Vec<Torus> = vec![0; messages.len()];
        for (pt, m) in plaintexts.iter_mut().zip(messages.iter()) {
            *pt = encoder.encode_core(*m)?;
        }
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

        let mut res = VectorLWE {
            ciphertexts: LweList::allocate(
                0,
                LweSize(sk.dimension + 1),
                CiphertextCount(messages.len()),
            ),
            variances: vec![0.; messages.len()],
            dimension: sk.dimension,
            nb_ciphertexts: messages.len(),
            encoders: vec![result_encoder; messages.len()],
        };
        res.encrypt_raw(sk, &plaintexts).unwrap();

        Ok(res)
    }

    /// Encode messages with a different encoder for each message and encrypt them
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `messages` -  a list of messages as u64
    /// * `encoders` - a list of Encoder elements
    ///
    /// # Output
    /// an VectorLWE structure
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // a list of encoders
    /// let encoders = [
    ///     Encoder::new(-2., 6., 4, 4).unwrap(),
    ///     Encoder::new(0., 3., 3, 2).unwrap(),
    ///     Encoder::new(4., 6., 2, 1).unwrap(),
    ///     Encoder::new(-1., 1., 5, 0).unwrap()];
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // two lists of messages
    /// let messages: Vec<f64> = vec![-1., 2., 5., -0.5];
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt_several_encoders(&secret_key, &messages, &encoders).unwrap();
    /// ```
    pub fn encode_encrypt_several_encoders(
        sk: &crate::LWESecretKey,
        messages: &[f64],
        encoders: &[crate::Encoder],
    ) -> Result<VectorLWE, CryptoAPIError> {
        let mut plaintexts: Vec<Torus> = vec![0; messages.len()];
        let mut result_encoders: Vec<crate::Encoder> = encoders.to_vec();
        for (((pt, m), enc), res_enc) in plaintexts
            .iter_mut()
            .zip(messages.iter())
            .zip(encoders.iter())
            .zip(result_encoders.iter_mut())
        {
            *pt = enc.encode_core(*m)?;

            let nb_bit_overlap: usize =
                res_enc.update_precision_from_variance(f64::powi(sk.std_dev, 2i32))?;

            // notification of a problem
            if nb_bit_overlap > 0 {
                println!(
                "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension the reduce the amount of noise needed.",
                "Loss of precision during encrypt".red().bold(),
                nb_bit_overlap, enc.nb_bit_precision
            );
            }
        }

        let mut res = VectorLWE {
            ciphertexts: LweList::allocate(
                0,
                LweSize(sk.dimension + 1),
                CiphertextCount(messages.len()),
            ),
            variances: vec![0.; messages.len()],
            dimension: sk.dimension,
            nb_ciphertexts: messages.len(),
            encoders: result_encoders,
        };
        res.encrypt_raw(sk, &plaintexts).unwrap();

        Ok(res)
    }

    /// Encrypt plaintexts from a Plaintext with the provided LWEParams
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintexts` - a list of plaintexts
    /// * `params` - LWE parameters
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
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
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let mut ct = VectorLWE::zero(sk.dimension, messages.len()).unwrap();
    /// ct.encrypt_inplace(&sk, &pt).unwrap();
    /// ```
    pub fn encrypt_inplace(
        &mut self,
        sk: &crate::LWESecretKey,
        plaintexts: &crate::Plaintext,
    ) -> Result<(), CryptoAPIError> {
        // encryption
        self.encrypt_raw(sk, &plaintexts.plaintexts).unwrap();

        for (output_enc, input_enc) in izip!(self.encoders.iter_mut(), plaintexts.encoders.iter()) {
            // copy the Encoders from the Plaintexts to the VectorLWE
            output_enc.copy(input_enc);

            // check if there is an overlap of the noise into the message
            let nb_bit_overlap: usize =
                output_enc.update_precision_from_variance(f64::powi(sk.std_dev, 2i32))?;

            // notification of a problem is there is overlap
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) with {} bit(s) of message originally. Consider increasing the dimension to reduce the amount of noise needed.",
                    "Loss of precision during encrypt".red().bold(),
                    nb_bit_overlap, input_enc.nb_bit_precision
                );
            }
        }
        Ok(())
    }

    /// Encrypt several raw plaintexts (list of Torus element instead of a struct Plaintext) with the provided key and standard deviation
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// * `plaintexts` - a list of plaintexts
    /// * `std_dev` - the standard deviation used for the error normal distribution
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // create an Encoder instance where messages are in the interval [-5, 5[
    /// let encoder = Encoder::new(-5., 5., 8, 0).unwrap();
    ///
    /// // create a list plaintexts
    /// let pt: Vec<u64> = vec![0; 5];
    ///
    /// // create one LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let mut ct = VectorLWE::zero(sk.dimension, pt.len()).unwrap();
    /// ct.encrypt_raw(&sk, &pt).unwrap();
    /// ```
    pub fn encrypt_raw(
        &mut self,
        sk: &crate::LWESecretKey,
        plaintexts: &[Torus],
    ) -> Result<(), CryptoAPIError> {
        // compute the variance
        let var = sk.get_variance();

        // check if we have enough std dev to have noise in the ciphertext
        if sk.std_dev < f64::powi(2., -(<Torus as Numeric>::BITS as i32) + 2) {
            return Err(NoNoiseInCiphertext!(var));
        }

        // fill the variance array
        for (self_var, _pt) in izip!(self.variances.iter_mut(), plaintexts.iter()) {
            *self_var = var;
        }

        // encrypt
        sk.val.encrypt_lwe_list(
            &mut self.ciphertexts,
            &PlaintextList::from_container(plaintexts),
            StandardDev::from_standard_dev(sk.std_dev),
            &mut EncryptionRandomGenerator::new(None),
        );

        Ok(())
    }

    /// Decrypt the list of ciphertexts, meaning compute the phase and directly decode the output
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// # Output
    /// * `result` - a list of messages as f64
    /// * DimensionError - if the ciphertext and the key have incompatible dimensions
    /// ```rust
    /// use concrete::*;
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
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let mut ct = VectorLWE::zero(sk.dimension, messages.len()).unwrap();
    /// ct.encrypt_inplace(&sk, &pt).unwrap();
    ///
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode(&self, sk: &crate::LWESecretKey) -> Result<Vec<f64>, CryptoAPIError> {
        // check dimensions
        if sk.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, sk.dimension));
        }

        // allocate the result
        let mut result: Vec<f64> = vec![0.; self.nb_ciphertexts];

        // create a temporary variable to store the result of the phase computation
        let mut tmp: Vec<Torus> = vec![0; self.nb_ciphertexts];

        // compute the phase
        sk.val.decrypt_lwe_list(
            &mut PlaintextList::from_container(tmp.as_mut_slice()),
            &self.ciphertexts,
        );

        // decode
        for (r, pt, enc) in izip!(result.iter_mut(), tmp.iter(), self.encoders.iter()) {
            *r = enc.decode_single(*pt)?;
        }

        Ok(result)
    }

    /// Decrypt the list of ciphertexts, meaning compute the phase and directly decode the output
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// # Output
    /// * `result` - a list of messages as f64
    /// * DimensionError - if the ciphertext and the key have incompatible dimensions
    /// ```rust
    /// use concrete::*;
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
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let mut ct = VectorLWE::zero(sk.dimension, messages.len()).unwrap();
    /// ct.encrypt_inplace(&sk, &pt).unwrap();
    ///
    /// let res = ct.decrypt_raw(&sk).unwrap();
    /// ```
    pub fn decrypt_raw(&self, sk: &crate::LWESecretKey) -> Result<Vec<u64>, CryptoAPIError> {
        // check dimensions
        if sk.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, sk.dimension));
        }

        // create a temporary variable to store the result of the phase computation
        let mut tmp: Vec<u64> = vec![0; self.nb_ciphertexts];

        // compute the phase
        sk.val.decrypt_lwe_list(
            &mut PlaintextList::from_container(tmp.as_mut_slice()),
            &self.ciphertexts,
        );

        Ok(tmp)
    }

    /// Decrypt the list of ciphertexts, meaning compute the phase and directly decode the output as if the encoder was set in round mode
    ///
    /// # Arguments
    /// * `sk` - an LWE secret key
    /// # Output
    /// * `result` - a list of messages as f64
    /// * DimensionError - if the ciphertext and the key have incompatible dimensions
    /// ```rust
    /// use concrete::*;
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
    /// // create an LWESecretKey
    /// let sk = LWESecretKey::new(&LWE128_630);
    ///
    /// // create a new VectorLWE that encrypts pt
    /// let mut ct = VectorLWE::zero(sk.dimension, messages.len()).unwrap();
    /// ct.encrypt_inplace(&sk, &pt).unwrap();
    ///
    /// let res = ct.decrypt_decode(&sk).unwrap();
    /// ```
    pub fn decrypt_decode_round(
        &self,
        sk: &crate::LWESecretKey,
    ) -> Result<Vec<f64>, CryptoAPIError> {
        // check dimensions
        if sk.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, sk.dimension));
        }

        // allocate the result
        let mut result: Vec<f64> = vec![0.; self.nb_ciphertexts];

        // create a temporary variable to store the result of the phase computation
        let mut tmp: Vec<Torus> = vec![0; self.nb_ciphertexts];

        // compute the phase
        sk.val.decrypt_lwe_list(
            &mut PlaintextList::from_container(tmp.as_mut_slice()),
            &self.ciphertexts,
        );

        // decode
        for (r, pt, enc) in izip!(result.iter_mut(), tmp.iter(), self.encoders.iter()) {
            let mut tmp_enc = enc.clone();
            tmp_enc.round = true;
            *r = tmp_enc.decode_single(*pt)?;
        }

        Ok(result)
    }

    /// Add small messages to a VectorLWE ciphertext and does not change the encoding but changes the bodies of the ciphertexts
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
    ///
    /// // encode and encrypt
    /// let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
    /// let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
    ///
    /// // addition between ciphertext and messages_2
    /// let ct_add = ciphertext.add_constant_static_encoder(&messages_2).unwrap();
    /// ```
    pub fn add_constant_static_encoder(
        &self,
        messages: &[f64],
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_static_encoder_inplace(messages)?;
        Ok(res)
    }

    /// Add small messages to a VectorLWE ciphertext and does not change the encoding but changes the bodies of the ciphertexts
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
    /// // two lists of messages
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
    ///
    /// // encode and encrypt
    /// let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
    /// let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
    ///
    /// // addition between ciphertext and messages_2
    /// ciphertext
    ///     .add_constant_static_encoder_inplace(&messages_2)
    ///     .unwrap();
    /// ```
    pub fn add_constant_static_encoder_inplace(
        &mut self,
        messages: &[f64],
    ) -> Result<(), CryptoAPIError> {
        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        for (mut ciphertext, lwe_encoder, m) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            self.encoders.iter(),
            messages.iter()
        ) {
            // error if one message is not in [-delta,delta]
            if m.abs() > lwe_encoder.delta {
                return Err(MessageTooBigError!(*m, lwe_encoder.delta));
            }
            let mut ec_tmp = lwe_encoder.clone();
            ec_tmp.o = 0.;
            let update = ciphertext
                .get_element(self.dimension)
                .wrapping_add(ec_tmp.encode_outside_interval_operators(*m)?);
            *ciphertext.get_element_mut(self.dimension) = update;
        }
        Ok(())
    }

    /// Add messages to a VectorLWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertexts
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
    ///
    /// // encode and encrypt
    /// let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
    /// let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
    ///
    /// // addition between ciphertext and messages_2
    /// let ct = ciphertext
    ///     .add_constant_dynamic_encoder(&messages_2)
    ///     .unwrap();
    /// ```
    pub fn add_constant_dynamic_encoder(
        &self,
        messages: &[f64],
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_constant_dynamic_encoder_inplace(messages)?;
        Ok(res)
    }

    /// Add messages to a VectorLWE ciphertext and translate the interval of a distance equal to the message but does not change either the bodies or the masks of the ciphertexts
    ///
    /// # Argument
    /// * `messages` - a list of messages as f64
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![-4.9, 1.02, 4.6, 5.6, -3.2];
    ///
    /// // encode and encrypt
    /// let plaintext_1 = Plaintext::encode(&messages_1, &encoder).unwrap();
    /// let mut ciphertext = VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();
    ///
    /// // addition between ciphertext and messages_2
    /// ciphertext
    ///     .add_constant_dynamic_encoder_inplace(&messages_2)
    ///     .unwrap();
    /// ```
    pub fn add_constant_dynamic_encoder_inplace(
        &mut self,
        messages: &[f64],
    ) -> Result<(), CryptoAPIError> {
        for (lwe_encoder, m) in izip!(self.encoders.iter_mut(), messages.iter()) {
            if !lwe_encoder.is_valid() {
                return Err(InvalidEncoderError!(
                    lwe_encoder.nb_bit_precision,
                    lwe_encoder.delta
                ));
            }
            lwe_encoder.o += m;
        }
        Ok(())
    }

    /// Compute an homomorphic addition between two VectorLWE ciphertexts
    ///
    /// # Arguments
    /// * `ct` - an VectorLWE struct
    /// * `new_min` - the min of the interval for the resulting Encoder
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // new_min
    /// let new_min: Vec<f64> = vec![103.; messages_1.len()];
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// let ct_add = ciphertext_1
    ///     .add_with_new_min(&ciphertext_2, &new_min)
    ///     .unwrap();
    /// ```
    pub fn add_with_new_min(
        &self,
        ct: &crate::VectorLWE,
        new_min: &[f64],
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_new_min_inplace(ct, new_min)?;
        Ok(res)
    }

    /// Compute an homomorphic addition between two VectorLWE ciphertexts
    ///
    /// # Arguments
    /// * `ct` - an VectorLWE struct
    /// * `new_min` - the min of the interval for the resulting Encoder
    ///
    /// # Output
    /// ** DimensionError - if the ciphertexts have incompatible dimensions
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// // new_min
    /// let new_min: Vec<f64> = vec![103.; messages_1.len()];
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// ciphertext_1
    ///     .add_with_new_min_inplace(&ciphertext_2, &new_min)
    ///     .unwrap();
    /// ```
    pub fn add_with_new_min_inplace(
        &mut self,
        ct: &crate::VectorLWE,
        new_min: &[f64],
    ) -> Result<(), CryptoAPIError> {
        // check dimensions
        if ct.dimension != self.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // add the two ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_add(ct.ciphertexts.as_tensor());

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        // correction related to the addition
        for (mut ciphertext, enc1, enc2, new) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            self.encoders.iter(),
            ct.encoders.iter(),
            new_min.iter()
        ) {
            // error if the deltas are not identical as well as the paddings
            if !deltas_eq!(enc1.delta, enc2.delta) {
                return Err(DeltaError!(enc1.delta, enc2.delta));
            } else if enc1.nb_bit_padding != enc2.nb_bit_padding {
                return Err(PaddingError!(enc1.nb_bit_padding, enc2.nb_bit_padding));
            }
            let mut tmp_ec = enc1.clone();
            tmp_ec.o = *new;
            let update = ciphertext
                .get_element(self.dimension)
                .wrapping_add(tmp_ec.encode_outside_interval_operators(enc1.o + enc2.o)?);
            *ciphertext.get_element_mut(self.dimension) = update;
        }

        // update the Encoder list
        for (enc, min) in self.encoders.iter_mut().zip(new_min.iter()) {
            enc.o = *min;
        }

        // update the noise with the NPE
        for (var1, var2, enc) in izip!(
            self.variances.iter_mut(),
            ct.variances.iter(),
            self.encoders.iter_mut()
        ) {
            *var1 = npe::add_ciphertexts(*var1, *var2);
            enc.update_precision_from_variance(*var1)?;
        }
        Ok(())
    }

    /// Compute an homomorphic addition between two VectorLWE ciphertexts.
    /// The center of the output Encoder is the sum of the two centers of the input Encoders.
    /// # Arguments
    /// * `ct` - an VectorLWE struct
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// let new_ciphertext = ciphertext_1.add_centered(&ciphertext_2).unwrap();
    /// ```
    pub fn add_centered(&self, ct: &crate::VectorLWE) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_centered_inplace(ct)?;
        Ok(res)
    }

    /// Compute an homomorphic addition between two VectorLWE ciphertexts.
    /// The center of the output Encoder is the sum of the two centers of the input Encoders
    ///
    /// # Arguments
    /// * `ct` - an VectorLWE struct
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// // addition between ciphertext_1 and ciphertext_2
    /// ciphertext_1.add_centered_inplace(&ciphertext_2).unwrap();
    /// ```
    pub fn add_centered_inplace(&mut self, ct: &crate::VectorLWE) -> Result<(), CryptoAPIError> {
        // check same dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // check same deltas
        for (self_enc, ct_enc) in self.encoders.iter_mut().zip(ct.encoders.iter()) {
            if !deltas_eq!(self_enc.delta, ct_enc.delta) {
                return Err(DeltaError!(self_enc.delta, ct_enc.delta));
            }
        }

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        // add ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_add(ct.ciphertexts.as_tensor());

        // correction related to the addition
        for (mut ciphertext, enc1) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            self.encoders.iter(),
        ) {
            let mut tmp_enc = enc1.clone();
            tmp_enc.o = 0.;
            let correction: Torus = tmp_enc.encode_core(enc1.delta / 2.)?;
            let update = ciphertext
                .get_element(self.dimension)
                .wrapping_sub(correction);
            *ciphertext.get_element_mut(self.dimension) = update;
        }

        // update the Encoder list and variances
        for (self_enc, ct_enc, self_var, ct_var) in izip!(
            self.encoders.iter_mut(),
            ct.encoders.iter(),
            self.variances.iter_mut(),
            ct.variances.iter()
        ) {
            // compute the new encoder
            self_enc.o += ct_enc.o + self_enc.delta / 2.;

            // compute the new variance
            *self_var = npe::add_ciphertexts(*self_var, *ct_var);

            // update the encoder precision based on the variance
            self_enc.update_precision_from_variance(*self_var)?;
        }

        Ok(())
    }

    /// Compute an addition between two VectorLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorLWE struct
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// let ct_add = ciphertext_1.add_with_padding(&ciphertext_2);
    /// ```
    pub fn add_with_padding(
        &self,
        ct: &crate::VectorLWE,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.add_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an addition between two VectorLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorLWE struct
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.add_with_padding_inplace(&ciphertext_2);
    /// ```
    pub fn add_with_padding_inplace(
        &mut self,
        ct: &crate::VectorLWE,
    ) -> Result<(), CryptoAPIError> {
        for (self_enc, ct_enc) in izip!(self.encoders.iter(), ct.encoders.iter()) {
            // check same paddings
            if self_enc.nb_bit_padding != ct_enc.nb_bit_padding {
                return Err(PaddingError!(
                    self_enc.nb_bit_padding,
                    ct_enc.nb_bit_padding
                ));
            }
            // check at least one bit of padding
            else if self_enc.nb_bit_padding == 0 {
                return Err(NotEnoughPaddingError!(self_enc.nb_bit_padding, 1));
            }
            // check same deltas
            else if !deltas_eq!(self_enc.delta, ct_enc.delta) {
                return Err(DeltaError!(self_enc.delta, ct_enc.delta));
            }
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // add ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_add(ct.ciphertexts.as_tensor());

        // update the Encoder list and variances
        for (self_enc, ct_enc, self_var, ct_var) in izip!(
            self.encoders.iter_mut(),
            ct.encoders.iter(),
            self.variances.iter_mut(),
            ct.variances.iter()
        ) {
            // compute the new variance
            *self_var = npe::add_ciphertexts(*self_var, *ct_var);

            // compute the new encoder
            self_enc.o += ct_enc.o;
            self_enc.delta *= 2.;
            self_enc.nb_bit_padding -= 1;
            self_enc.nb_bit_precision =
                usize::min(self_enc.nb_bit_precision, ct_enc.nb_bit_precision);
            // update the encoder precision based on the variance
            self_enc.update_precision_from_variance(*self_var)?;
        }
        Ok(())
    }

    /// Compute an subtraction between two VectorLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorLWE struct
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// let ct_sub = ciphertext_1.add_with_padding(&ciphertext_2);
    /// ```
    pub fn sub_with_padding(
        &self,
        ct: &crate::VectorLWE,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.sub_with_padding_inplace(ct)?;
        Ok(res)
    }

    /// Compute an subtraction between two VectorLWE ciphertexts by eating one bit of padding
    ///
    /// # Argument
    /// * `ct` - an VectorLWE struct
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
    /// let messages_1: Vec<f64> = vec![106.276, 104.3, 100.12, 101.1, 107.78];
    /// let messages_2: Vec<f64> = vec![4.9, 3.02, 4.6, 2.6, 3.2];
    ///
    /// // encode and encrypt
    /// let mut ciphertext_1 = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    /// let ciphertext_2 = VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();
    ///
    /// ciphertext_1.sub_with_padding_inplace(&ciphertext_2);
    /// ```
    pub fn sub_with_padding_inplace(
        &mut self,
        ct: &crate::VectorLWE,
    ) -> Result<(), CryptoAPIError> {
        for (self_enc, ct_enc) in izip!(self.encoders.iter(), ct.encoders.iter()) {
            // check same paddings
            if self_enc.nb_bit_padding != ct_enc.nb_bit_padding {
                return Err(PaddingError!(
                    self_enc.nb_bit_padding,
                    ct_enc.nb_bit_padding
                ));
            }
            // check at least one bit of padding
            else if self_enc.nb_bit_padding == 0 {
                return Err(NotEnoughPaddingError!(self_enc.nb_bit_padding, 1));
            }
            // check same deltas
            else if !deltas_eq!(self_enc.delta, ct_enc.delta) {
                return Err(DeltaError!(self_enc.delta, ct_enc.delta));
            }
        }

        // check the dimensions
        if self.dimension != ct.dimension {
            return Err(DimensionError!(self.dimension, ct.dimension));
        }

        // subtract ciphertexts together
        self.ciphertexts
            .as_mut_tensor()
            .update_with_wrapping_sub(ct.ciphertexts.as_tensor());

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        // correction related to the subtraction
        for (mut ciphertext, enc1) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            self.encoders.iter(),
        ) {
            let correction: Torus = 1 << (<Torus as Numeric>::BITS - enc1.nb_bit_padding);
            let update = ciphertext
                .get_element(self.dimension)
                .wrapping_add(correction);
            *ciphertext.get_element_mut(self.dimension) = update;
        }

        // update the Encoder list
        for (self_enc, ct_enc) in self.encoders.iter_mut().zip(ct.encoders.iter()) {
            self_enc.o -= ct_enc.o + ct_enc.delta;
            self_enc.delta *= 2.;
            self_enc.nb_bit_padding -= 1;
        }

        // update the noise with the NPE
        for (var1, var2, enc) in izip!(
            self.variances.iter_mut(),
            ct.variances.iter(),
            self.encoders.iter_mut()
        ) {
            *var1 = npe::add_ciphertexts(*var1, *var2);
            enc.update_precision_from_variance(*var1)?;
        }
        Ok(())
    }

    /// Multiply VectorLWE ciphertexts with small integer messages and does not change the encoding but changes the bodies and masks of the ciphertexts
    ///
    /// # Argument
    /// * `messages` - a list of integer messages
    ///
    /// # Output
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![6.923, 3.70, 1.80, 0.394, -7.09];
    /// let messages_2: Vec<i32> = vec![2, 3, 5, -2, 0];
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    /// let new_ciphertext = ciphertext.mul_constant_static_encoder(&messages_2).unwrap();
    /// ```
    pub fn mul_constant_static_encoder(
        &self,
        messages: &[i32],
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.mul_constant_static_encoder_inplace(messages)?;
        Ok(res)
    }

    /// Multiply VectorLWE ciphertexts with small integer messages and does not change the encoding but changes the bodies and masks of the ciphertexts
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
    /// let messages_1: Vec<f64> = vec![6.923, 3.70, 1.80, 0.394, -7.09];
    /// let messages_2: Vec<i32> = vec![2, 3, 5, -2, 0];
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    /// ciphertext
    ///     .mul_constant_static_encoder_inplace(&messages_2)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_static_encoder_inplace(
        &mut self,
        messages: &[i32],
    ) -> Result<(), CryptoAPIError> {
        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        for (mut ciphertext, m, encoder, var) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            messages.iter(),
            self.encoders.iter_mut(),
            self.variances.iter_mut(),
        ) {
            // compute correction
            let cor0: Torus = encoder.encode_outside_interval_operators(0.)?;
            let cor = cor0.wrapping_mul((*m - 1) as Torus);

            // multiplication
            ciphertext.update_with_wrapping_scalar_mul(&(*m as Torus));

            // apply correction
            let update = ciphertext.get_element(self.dimension).wrapping_sub(cor);
            *ciphertext.get_element_mut(self.dimension) = update;

            // compute the absolute value
            let m_abs = m.abs();
            // call to the NPE to estimate the new variance
            *var = npe::LWE::single_scalar_mul(*var, m_abs as Torus);

            if m_abs != 0 {
                // update the encoder precision based on the variance
                encoder.update_precision_from_variance(*var)?;
            }
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
    /// * a new VectorLWE
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
    /// let messages_1: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
    /// let messages_2: Vec<f64> = vec![2.432, 3.87, 5.27, -2.13, 0.56];
    /// let b: f64 = 6.;
    ///
    /// // encode and encrypt
    /// let ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    /// let new_ciphertext = ciphertext
    ///     .mul_constant_with_padding(&messages_2, b, precision)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_with_padding(
        &self,
        constants: &[f64],
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.mul_constant_with_padding_inplace(constants, max_constant, nb_bit_padding)?;
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
    /// let messages_1: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
    /// let messages_2: Vec<f64> = vec![2.432, 3.87, 5.27, -2.13, 0.56];
    /// let b: f64 = 6.;
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    /// ciphertext
    ///     .mul_constant_with_padding_inplace(&messages_2, b, precision)
    ///     .unwrap();
    /// ```
    pub fn mul_constant_with_padding_inplace(
        &mut self,
        constants: &[f64],
        max_constant: f64,
        nb_bit_padding: usize,
    ) -> Result<(), CryptoAPIError> {
        // check if we have the same number of messages and ciphertexts
        if constants.len() != self.nb_ciphertexts {
            return Err(NbCTError!(constants.len(), self.nb_ciphertexts));
        }

        // some checks
        for (c, encoder) in izip!(constants.iter(), self.encoders.iter_mut()) {
            // check that the constant if below the maximum
            if *c > max_constant || *c < -max_constant {
                return Err(ConstantMaximumError!(*c, max_constant));
            }
            // check that zero is in the interval
            else if encoder.o > 0. || encoder.o + encoder.delta < 0. {
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

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        for (mut ciphertext, c, encoder, var) in izip!(
            self.ciphertexts.as_mut_tensor().subtensor_iter_mut(ct_size),
            constants.iter(),
            self.encoders.iter_mut(),
            self.variances.iter_mut(),
        ) {
            // test if negative
            let negative: bool = *c < 0.;

            // absolute value
            let c_abs = c.abs();

            // discretize c_abs with regard to the number of bits of padding to use
            let scal: Torus =
                (c_abs / max_constant * f64::powi(2., nb_bit_padding as i32)).round() as Torus;

            // encode 0 and subtract it
            let tmp_sub = encoder.encode_core(0.)?;
            let update = ciphertext.get_element(self.dimension).wrapping_sub(tmp_sub);
            *ciphertext.get_element_mut(self.dimension) = update;

            // scalar multiplication
            ciphertext.update_with_wrapping_scalar_mul(&scal);

            // new encoder
            let new_o = encoder.o * max_constant;
            let new_max = (encoder.o + encoder.delta - encoder.get_granularity()) * max_constant;
            let new_delta = new_max - new_o;

            // compute the discretization of c_abs
            let discret_c_abs =
                (scal as f64) * f64::powi(2., -(nb_bit_padding as i32)) * max_constant;

            // compute  the rounding error on c_abs
            let rounding_error = (discret_c_abs - c_abs).abs();

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
            // call to the NPE to estimate the new variance
            *var = npe::LWE::single_scalar_mul(*var, scal);

            if scal != 0 {
                // update the encoder precision based on the variance
                encoder.update_precision_from_variance(*var)?;
            }

            // encode 0 with the new encoder
            let tmp_add = encoder.encode_core(0.)?;
            let update = ciphertext.get_element(self.dimension).wrapping_add(tmp_add);
            *ciphertext.get_element_mut(self.dimension) = update;

            if negative {
                // compute the opposite
                ciphertext.update_with(|a| *a = a.wrapping_neg());

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
                let update = ciphertext.get_element(self.dimension).wrapping_add(cor);
                *ciphertext.get_element_mut(self.dimension) = update;

                // change the encoder
                encoder.opposite_inplace()?;
            }
        }
        Ok(())
    }

    /// Compute the opposite of the n-th LWE ciphertext in the structure
    ///
    /// # Argument
    /// * `n` - index of a LWE ciphertext
    ///
    /// # Output
    /// * a new VectorLWE ciphertext
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
    /// let messages_1: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
    ///
    /// // encode and encrypt
    /// let ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    ///
    /// let new_ciphertext = ciphertext.opposite_nth(3).unwrap();
    /// ```
    pub fn opposite_nth(&self, n: usize) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut res = self.clone();
        res.opposite_nth_inplace(n)?;
        Ok(res)
    }

    /// Compute the opposite of the n-th LWE ciphertext in the structure
    ///
    /// # Argument
    /// * `n` - index of a LWE ciphertext
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
    /// let messages_1: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
    ///
    /// // encode and encrypt
    /// let mut ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder).unwrap();
    ///
    /// ciphertext.opposite_nth_inplace(3).unwrap();
    /// ```
    pub fn opposite_nth_inplace(&mut self, n: usize) -> Result<(), CryptoAPIError> {
        // check the index n
        if n >= self.nb_ciphertexts {
            return Err(IndexError!(self.nb_ciphertexts, n));
        }
        // check the encoders
        else if !self.encoders[n].is_valid() {
            return Err(InvalidEncoderError!(
                self.encoders[n].nb_bit_precision,
                self.encoders[n].delta
            ));
        }

        // get the size of one lwe ciphertext
        let ct_size = self.get_ciphertext_size();

        // select the n-th ciphertext
        let mut ct = self
            .ciphertexts
            .as_mut_tensor()
            .get_sub_mut((n * (ct_size))..((n + 1) * (ct_size)));

        // compute the opposite
        ct.update_with(|a| *a = a.wrapping_neg());

        // add correction if there is some padding
        let mut cor: Torus = 0;
        if self.encoders[n].nb_bit_padding > 0 {
            cor = (1 << (<Torus as Numeric>::BITS - self.encoders[n].nb_bit_padding))
                - (1 << (<Torus as Numeric>::BITS
                    - self.encoders[n].nb_bit_padding
                    - self.encoders[n].nb_bit_precision));
        } else {
            cor = cor.wrapping_sub(
                1 << (<Torus as Numeric>::BITS
                    - self.encoders[n].nb_bit_padding
                    - self.encoders[n].nb_bit_precision),
            );
        }
        let update = ct.get_element(self.dimension).wrapping_add(cor);
        *ct.get_element_mut(self.dimension) = update;

        // change the encoder
        self.encoders[n].opposite_inplace()?;

        Ok(())
    }

    /// Compute a key switching operation on every ciphertext from the VectorLWE struct self
    ///
    /// # Argument
    /// * `ksk` - the key switching key
    ///
    /// # Output
    /// * a VectorLWE struct
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
    /// let messages: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
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
    ///     VectorLWE::encode_encrypt(&secret_key_before, &messages, &encoder).unwrap();
    ///
    /// // key switch
    /// let ciphertext_after = ciphertext_before.keyswitch(&ksk).unwrap();
    /// ```
    pub fn keyswitch(&self, ksk: &crate::LWEKSK) -> Result<crate::VectorLWE, CryptoAPIError> {
        // allocation for the result
        let mut res: crate::VectorLWE =
            crate::VectorLWE::zero(ksk.dimension_after, self.nb_ciphertexts)?;

        // key switch
        ksk.ciphertexts
            .keyswitch_list(&mut res.ciphertexts, &self.ciphertexts);

        // deal with encoders, noise and new precision
        for (output_enc, input_enc, vout, vin) in izip!(
            res.encoders.iter_mut(),
            self.encoders.iter(),
            res.variances.iter_mut(),
            self.variances.iter()
        ) {
            // calls the NPE to find out the amount of noise after KS
            *vout = <Torus as LWE>::key_switch(
                self.dimension,
                ksk.level,
                ksk.base_log,
                ksk.variance,
                *vin,
            );

            // copy the encoders
            output_enc.copy(input_enc);

            // update the precision
            let nb_bit_overlap: usize = output_enc.update_precision_from_variance(*vout)?;

            // notification of a problem
            if nb_bit_overlap > 0 {
                println!(
                    "{}: {} bit(s) lost, with {} bit(s) of message originally",
                    "Loss of precision during key switch".red().bold(),
                    nb_bit_overlap,
                    input_enc.nb_bit_precision
                );
            }
        }

        Ok(res)
    }

    /// Compute a bootstrap on the n-th LWE from the self VectorLWE structure
    ///
    /// # Argument
    /// * `bsk` - the bootstrapping key
    /// * `n` - the index of the ciphertext to bootstrap
    ///
    /// # Output
    /// * a VectorLWE struct
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
    /// // two lists of messages
    /// let messages: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
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
    /// // a list of messages that we encrypt
    /// let ciphertext_before =
    ///     VectorLWE::encode_encrypt(&secret_key_before, &messages, &encoder).unwrap();
    ///
    /// let ciphertext_out = ciphertext_before
    ///     .bootstrap_nth(&bootstrapping_key, 2)
    ///     .unwrap();
    /// ```
    pub fn bootstrap_nth(
        &self,
        bsk: &crate::LWEBSK,
        n: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        self.bootstrap_nth_with_function(bsk, |x| x, &self.encoders[n], n)
    }

    /// Compute a bootstrap and apply an arbitrary function to the given VectorLWE ciphertext
    ///
    /// # Argument
    /// * `bsk` - the bootstrapping key
    /// * `f` - the function to apply
    /// * `encoder_output` - a list of output encoders
    /// * `n` - the index of the ciphertext to bootstrap
    ///
    /// # Output
    /// * a VectorLWE struct
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
    /// // a list of messages
    /// let messages: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1, -107.78];
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
    /// let ciphertext_before =
    ///     VectorLWE::encode_encrypt(&secret_key_before, &messages, &encoder_input).unwrap();
    ///
    /// let ciphertext_out = ciphertext_before
    ///     .bootstrap_nth_with_function(&bootstrapping_key, |x| f64::max(0., x), &encoder_output, 2)
    ///     .unwrap();
    /// ```
    pub fn bootstrap_nth_with_function<F: Fn(f64) -> f64>(
        &self,
        bsk: &crate::LWEBSK,
        f: F,
        encoder_output: &crate::Encoder,
        n: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        // check the index n
        if n >= self.nb_ciphertexts {
            return Err(IndexError!(self.nb_ciphertexts, n));
        }
        // check bsk compatibility
        if self.dimension != bsk.get_lwe_dimension() {
            return Err(DimensionError!(self.dimension, bsk.get_lwe_dimension()));
        }

        // generate the look up table
        let lut = bsk.generate_functional_look_up_table(&self.encoders[n], encoder_output, f)?;

        // build the trivial accumulator
        let mut accumulator = GlweCiphertext::allocate(
            0,
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

        if self.encoders[n].nb_bit_padding > 1 {
            // copy the ciphertext to bootstrap
            let mut ct_clone = self
                .ciphertexts
                .as_tensor()
                .get_sub(n * (self.get_ciphertext_size())..((n + 1) * (self.get_ciphertext_size())))
                .iter()
                .copied()
                .collect::<Vec<Torus>>();

            // shift of some bits to the left
            Tensor::from_container(ct_clone.as_mut_slice())
                .update_with_scalar_shl(&(self.encoders[n].nb_bit_padding - 1));

            // compute the bootstrap
            bsk.ciphertexts.bootstrap(
                &mut result,
                &LweCiphertext::from_container(ct_clone),
                & accumulator,
            );
        } else {
            // compute the bootstrap
            let ct_view = self
                .ciphertexts
                .as_tensor()
                .get_sub(n * (self.get_ciphertext_size())..((n + 1) * (self.get_ciphertext_size())))
                .into_container();
            let ct = LweCiphertext::from_container(ct_view);
            bsk.ciphertexts
                .bootstrap(&mut result, &ct, & accumulator);
        }

        // compute the new variance (without the drift)
        let new_var = <Torus as npe::Cross>::bootstrap(
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
                nb_bit_overlap, self.encoders[n].nb_bit_precision
            );
        }

        // calls the NPE to find out the amount of noise after rounding the input ciphertext (drift)
        let nb_rounding_noise_bit: usize =
            (npe::lwe::log2_rounding_noise(self.dimension)).ceil() as usize + 1;

        // deals with the drift error
        if nb_rounding_noise_bit
            + self.encoders[n].nb_bit_padding
            + new_encoder_output.nb_bit_precision
            > bsk.get_polynomial_size_log() + 1
        {
            let nb_bit_loss = self.encoders[n].nb_bit_padding
                + new_encoder_output.nb_bit_precision
                + nb_rounding_noise_bit
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
                nb_bit_loss, self.encoders[n].nb_bit_precision,nb_rounding_noise_bit
            );
        }

        // construct the output
        let lwe = crate::VectorLWE {
            variances: vec![new_var; 1],
            ciphertexts: LweList::from_container(
                result.into_tensor().into_container(),
                LweSize(bsk.polynomial_size * bsk.dimension + 1),
            ),
            dimension: bsk.polynomial_size * bsk.dimension,
            nb_ciphertexts: 1,
            encoders: vec![new_encoder_output; 1],
        };

        Ok(lwe)
    }

    /// Multiply two LWE ciphertexts thanks to two bootstrapping procedures
    /// need to have 2 bits of padding at least
    ///
    /// # Argument
    /// * `ct` - an VectorLWE struct containing the second LWE for the multiplication
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
    /// let messages_1: Vec<f64> = vec![-127., -36.2, 58.7, 161.1, 69.1];
    /// let messages_2: Vec<f64> = vec![72.7, 377., 59.6, 115.5, 286.3];
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
    ///     VectorLWE::encode_encrypt(&secret_key_before, &messages_1, &encoder_1).unwrap();
    ///
    /// let ciphertext_2 =
    ///     VectorLWE::encode_encrypt(&secret_key_before, &messages_2, &encoder_2).unwrap();
    ///
    /// let ciphertext_out = ciphertext_1
    ///     .mul_from_bootstrap_nth(&ciphertext_2, &bootstrapping_key, 2, 3)
    ///     .unwrap();
    /// ```
    pub fn mul_from_bootstrap_nth(
        &self,
        ct: &crate::VectorLWE,
        bsk: &crate::LWEBSK,
        n_self: usize,
        n_ct: usize,
    ) -> Result<crate::VectorLWE, CryptoAPIError> {
        // extract twice from self
        let mut ct1 = self.extract_nth(n_self)?;
        let mut ct2 = self.extract_nth(n_self)?;

        // return an error if nb_bit_precision < 2
        if ct1.encoders[0].nb_bit_precision < 2 {
            return Err(NotEnoughPaddingError!(ct1.encoders[0].nb_bit_precision, 2));
        }

        // extract once from ct

        let input2 = ct.extract_nth(n_ct)?;

        // compute the addition and the subtraction
        ct1.add_with_padding_inplace(&input2)?;
        ct2.sub_with_padding_inplace(&input2)?;

        // create the output encoder
        let mut encoder_out1 = ct1.encoders[0].new_square_divided_by_four(2)?;
        let mut encoder_out2 = ct2.encoders[0].new_square_divided_by_four(2)?;

        // set the same deltas
        if encoder_out1.delta < encoder_out2.delta {
            encoder_out1.delta = encoder_out2.delta;
        } else {
            encoder_out2.delta = encoder_out1.delta;
        }

        // bootstrap
        let mut square1 =
            ct1.bootstrap_nth_with_function(bsk, |x| (x * x) / 4., &encoder_out1, 0)?;
        let square2 = ct2.bootstrap_nth_with_function(bsk, |x| (x * x) / 4., &encoder_out2, 0)?;

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

    pub fn load(path: &str) -> Result<VectorLWE, Box<dyn Error>> {
        read_from_file(path)
    }

    pub fn pp(&self) {
        for (variance, encoder) in izip!(self.variances.iter(), self.encoders.iter()) {
            let mut padding_res: String = "".to_string();
            // padding part
            for _i in 0..encoder.nb_bit_padding {
                padding_res.push('o');
            }
            let mut message_res: String = "".to_string();

            // message part
            for _i in 0..encoder.nb_bit_precision {
                message_res.push('o');
            }
            let noise = npe::nb_bit_from_variance_99(*variance, <Torus as Numeric>::BITS as usize);
            // <Torus as Types>::TORUS_BIT  + (f64::log2(3. * f64::sqrt(*variance))).floor() as usize;
            let mut noise_res: String = "".to_string();
            // nose part
            for _i in 0..noise {
                noise_res.push('o');
            }
            assert!(
                noise <= 64,
                "noise = {} ; {}",
                noise,
                (f64::log2(3. * f64::sqrt(*variance))).floor()
            );

            let nothing = <Torus as Numeric>::BITS
                - encoder.nb_bit_padding
                - encoder.nb_bit_precision
                - noise;
            assert!(nothing <= 64, "nothing = {} ", nothing);
            let mut nothing_res: String = "".to_string();
            // nth part
            for _i in 0..nothing {
                nothing_res.push('o');
            }

            println!(
                "{}{}{}{}",
                padding_res.blue().bold(),
                message_res.green().bold(),
                nothing_res.bold(),
                noise_res.red().bold()
            );
        }
        println!();
    }

    /// Sum all the LWE ciphertexts contained in self into one single ciphertext and output it as a new VectorLWE
    ///
    /// # Output
    /// * A new VectorLWE containing only one ciphertext
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 4;
    /// let padding = 2;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a list of messages
    /// let messages: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1];
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a list of messages that we encrypt
    /// let ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
    ///
    /// // sum
    /// let ciphertext_sum = ciphertext.sum_with_padding().unwrap();
    /// ```
    pub fn sum_with_padding(&self) -> Result<crate::VectorLWE, CryptoAPIError> {
        let nb_bit_padding_consumed: usize =
            f64::ceil(f64::log2(self.nb_ciphertexts as f64)) as usize;
        let mut ct = Tensor::from_container(vec![0; self.dimension + 1]);
        let mut new_var: f64 = 0.;
        let mut new_o: f64 = 0.;
        let new_delta: f64 = self.encoders[0].delta * f64::powi(2., nb_bit_padding_consumed as i32);
        let mut new_precision: usize = self.encoders[0].nb_bit_precision;

        for (ct_in, var_in, enc_in) in izip!(
            self.ciphertexts
                .as_tensor()
                .subtensor_iter(self.dimension + 1),
            self.variances.iter(),
            self.encoders.iter()
        ) {
            //Find the minimum precision among all the ciphertexts
            new_precision = usize::min(new_precision, enc_in.nb_bit_precision);
            //Check same deltas and paddings among all the ciphertexts
            if !deltas_eq!(self.encoders[0].delta, enc_in.delta) {
                return Err(DeltaError!(self.encoders[0].delta, enc_in.delta));
            } else if self.encoders[0].nb_bit_padding != enc_in.nb_bit_padding {
                return Err(PaddingError!(
                    self.encoders[0].nb_bit_padding,
                    enc_in.nb_bit_padding
                ));
            }
            ct.update_with_wrapping_add(&ct_in);
            new_var = npe::add_ciphertexts(new_var, *var_in);
            new_o += enc_in.o;
        }
        let mut new_encoder: crate::Encoder = crate::Encoder {
            o: new_o,
            delta: new_delta,
            nb_bit_precision: self.encoders[0].nb_bit_precision,
            nb_bit_padding: self.encoders[0].nb_bit_padding - nb_bit_padding_consumed,
            round: self.encoders[0].round,
        };
        new_encoder.update_precision_from_variance(new_var)?;
        Ok(VectorLWE {
            ciphertexts: LweList::from_container(ct.into_container(), LweSize(self.dimension + 1)),
            variances: vec![new_var],
            dimension: self.dimension,
            nb_ciphertexts: 1,
            encoders: vec![new_encoder],
        })
    }

    /// Sum all the LWE ciphertexts contained in self into one single ciphertext and output it as a
    /// new VectorLWE. The output ciphertext will have an encoder with the same size so we need to
    /// provide the min of the output interval
    ///
    /// # Input
    /// * `new_min` - an f64 containing the min of the output encoder
    ///
    /// # Output
    /// * A new VectorLWE containing only one ciphertext
    ///
    /// # Example
    /// ```rust
    /// use concrete::*;
    ///
    /// // params
    /// let (min, max): (f64, f64) = (-150., 204.);
    /// let precision = 4;
    /// let padding = 0;
    /// let level: usize = 3;
    /// let base_log: usize = 3;
    ///
    /// // encoder
    /// let encoder = Encoder::new(min, max, precision, padding).unwrap();
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a list of messages
    /// let messages: Vec<f64> = vec![-106.276, 104.3, -100.12, 101.1];
    ///
    /// // generate a secret key
    /// let secret_key = LWESecretKey::new(&LWE128_1024);
    ///
    /// // a list of messages that we encrypt
    /// let ciphertext = VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();
    ///
    /// // sum with a new min
    /// let ciphertext_sum = ciphertext.sum_with_new_min(-50.).unwrap();
    /// ```
    pub fn sum_with_new_min(&self, new_min: f64) -> Result<crate::VectorLWE, CryptoAPIError> {
        let mut ct = LweList::allocate(0, LweSize(self.dimension + 1), CiphertextCount(1));

        // add the ciphertexts together
        for ct_in in izip!(self
            .ciphertexts
            .as_tensor()
            .subtensor_iter(self.dimension + 1))
        {
            ct.as_mut_tensor().update_with_wrapping_add(&ct_in);
        }

        // deal with the resulting encoding and correcting terms
        let mut new_encoder: crate::Encoder = self.encoders[0].clone();
        new_encoder.o = new_min;
        let mut sum_min = 0.;
        for enc_in in self.encoders.iter() {
            sum_min += enc_in.o;
        }
        let correction = new_encoder.encode_outside_interval_operators(sum_min)?;
        let update = ct
            .as_tensor()
            .get_element(self.dimension)
            .wrapping_add(correction);
        *ct.as_mut_tensor().get_element_mut(self.dimension) = update;

        // deal with the new variance
        let new_var = npe::add_several_ciphertexts(&self.variances);
        new_encoder.update_precision_from_variance(new_var)?;

        // build the output
        Ok(VectorLWE {
            ciphertexts: ct,
            variances: vec![new_var],
            dimension: self.dimension,
            nb_ciphertexts: 1,
            encoders: vec![new_encoder],
        })
    }
}

/// Print needed pieces of information about an VectorLWE
impl fmt::Display for VectorLWE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = 2;
        let mut to_be_print: String = "".to_string();

        to_be_print += " VectorLWE {\n         -> samples = [";

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
            for elt in self.variances[self.variances.len() - n..].iter() {
                to_be_print = to_be_print + &format!("{}, ", elt);
            }
        }
        to_be_print += "]\n";
        to_be_print = to_be_print + &format!("         -> dimension = {}\n", self.dimension);
        to_be_print =
            to_be_print + &format!("         -> nb of ciphertexts = {}\n", self.nb_ciphertexts);
        to_be_print += "       }";
        writeln!(f, "{}", to_be_print)
    }
}
