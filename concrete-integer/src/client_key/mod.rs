//! This module implements the generation of the client secret ShortIntegerClientKeykeys, together
//! with the encryption and decryption methods.

pub(crate) mod utils;

use crate::ciphertext::Ciphertext;
use crate::client_key::utils::i_crt;
use concrete_shortint::parameters::MessageModulus;
use serde::{Deserialize, Serialize};
pub use utils::radix_decomposition;

/// The number of ciphertexts in the vector.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct VecLength(pub usize);

/// A structure containing the client key, which must be kept secret.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ClientKey {
    pub(crate) key: concrete_shortint::client_key::ClientKey,
    pub(crate) vector_length: VecLength,
}

impl ClientKey {
    /// Allocates and generates a client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // Generate the client key associated to integers over 4 blocks
    /// // of messages with modulus over 2 bits
    /// let cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 4);
    /// ```
    pub fn new(parameter_set: concrete_shortint::parameters::Parameters, size: usize) -> Self {
        Self {
            key: concrete_shortint::client_key::ClientKey::new(parameter_set),
            vector_length: VecLength(size),
        }
    }

    /// Allocates and generates a client key from an existing shortint client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::{ClientKey, VecLength};
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let shortint_cks = concrete_shortint::ClientKey::new(PARAM_MESSAGE_2_CARRY_2);
    /// let cks = ClientKey::from_shortint(shortint_cks, VecLength(4));
    /// ```
    pub fn from_shortint(
        key: concrete_shortint::client_key::ClientKey,
        vector_length: VecLength,
    ) -> Self {
        Self { key, vector_length }
    }

    /// Returns the parameters used by the client key.
    pub fn parameters(&self) -> concrete_shortint::Parameters {
        self.key.parameters
    }

    /// Encrypts a integer message using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let mut cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 4);
    ///
    /// let msg = 12_u64;
    ///
    /// // Encryption of one message:
    /// let ct = cks.encrypt(msg);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt(&self, message: u64) -> Ciphertext {
        let mut ct_vec: Vec<concrete_shortint::ciphertext::Ciphertext> = Vec::new();
        let mut message_modulus_vec: Vec<u64> = Vec::new();

        //Bits of message put to 1
        let mask = (self.key.parameters.message_modulus.0 - 1) as u64;

        let mut power = 1_u64;
        //Put each decomposition into a new ciphertext
        for _ in 0..self.vector_length.0 {
            let mut decomp = message & (mask * power);
            decomp /= power;

            // encryption
            let ct = self.key.encrypt(decomp);
            ct_vec.push(ct);

            // fill the vector with the message moduli
            message_modulus_vec.push(self.key.parameters.message_modulus.0 as u64);

            //modulus to the power i
            power *= self.key.parameters.message_modulus.0 as u64;
        }

        Ciphertext {
            ct_vec,
            message_modulus_vec,
            key_id_vec: vec![],
        }
    }

    /// Encrypts a integer message using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let mut cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 4);
    ///
    /// let msg = 12_u64;
    ///
    /// // Encryption of one message:
    /// let ct = cks.encrypt_without_padding(msg);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_without_padding(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt_without_padding(&self, message: u64) -> Ciphertext {
        // Empty vector of ciphertexts
        let mut ct_vec = Vec::<concrete_shortint::Ciphertext>::with_capacity(self.vector_length.0);
        let mut message_modulus_vec: Vec<u64> = Vec::<u64>::with_capacity(self.vector_length.0);

        // Bits of message put to 1
        let mask = (self.key.parameters.message_modulus.0 - 1) as u64;

        let mut power = 1_u64;
        // Put each decomposition into a new ciphertext
        for _ in 0..self.vector_length.0 {
            let mut decomp = message & (mask * power);
            decomp /= power;

            // encryption
            let ct = self.key.encrypt_without_padding(decomp);
            ct_vec.push(ct);

            // fill the base vector
            message_modulus_vec.push(self.key.parameters.message_modulus.0 as u64);

            // modulus to the power i
            power *= self.key.parameters.message_modulus.0 as u64;
        }

        Ciphertext {
            ct_vec,
            message_modulus_vec,
            key_id_vec: vec![],
        }
    }

    /// Encrypts one block.
    ///
    /// This returns a shortint ciphertext.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let mut cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 4);
    ///
    /// let msg = 2_u64;
    ///
    /// // Encryption of one message:
    /// let ct = cks.encrypt_one_block(msg);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_one_block(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt_one_block(&self, message: u64) -> concrete_shortint::Ciphertext {
        self.key.encrypt(message)
    }

    /// Decrypts one block.
    ///
    /// This takes a shortint ciphertext as input.
    pub fn decrypt_one_block(&self, ct: &concrete_shortint::Ciphertext) -> u64 {
        self.key.decrypt(ct)
    }

    /// Decrypts a ciphertext encrypting an integer message + carries using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::DEFAULT_PARAMETERS;
    ///
    /// let cks = ClientKey::new(DEFAULT_PARAMETERS, 4);
    ///
    /// let msg = 12_u64;
    ///
    /// // Encryption of one message:
    /// let ct = cks.encrypt(msg);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt(&self, ctxt: &Ciphertext) -> u64 {
        let mut result = 0_u64;
        let mut shift = 1_u64;
        for (c_i, b_i) in ctxt.ct_vec.iter().zip(ctxt.message_modulus_vec.iter()) {
            //decrypt the component i of the integer and multiply it by the radix product
            let tmp = self.key.decrypt_message_and_carry(c_i).wrapping_mul(shift);

            // update the result
            result = result.wrapping_add(tmp);

            // update the shift for the next iteration
            shift = shift.wrapping_mul(*b_i);
        }

        let mut product = 1_u64;
        for b_i in ctxt.message_modulus_vec.iter() {
            product = product.wrapping_mul(*b_i);
        }

        result % product
    }

    /// Decrypts a ciphertext encrypting an integer message + carries using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::DEFAULT_PARAMETERS;
    ///
    /// let cks = ClientKey::new(DEFAULT_PARAMETERS, 4);
    ///
    /// let msg = 12_u64;
    ///
    /// // Encryption of one message:
    /// let ct = cks.encrypt(msg);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt_without_padding(&self, ctxt: &Ciphertext) -> u64 {
        let mut result = 0_u64;
        let mut shift = 1_u64;
        for (c_i, b_i) in ctxt.ct_vec.iter().zip(ctxt.message_modulus_vec.iter()) {
            // decrypt the component i of the integer and multiply it by the radix product
            let tmp = self
                .key
                .decrypt_message_and_carry_without_padding(c_i)
                .wrapping_mul(shift);

            // update the result
            result = result.wrapping_add(tmp);

            // update the shift for the next iteration
            shift = shift.wrapping_mul(*b_i);
        }

        let mut product = 1_u64;
        for b_i in ctxt.message_modulus_vec.iter() {
            product = product.wrapping_mul(*b_i);
        }

        result % product
    }

    /// Encrypts a small integer message using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// let cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 3);
    ///
    /// let msg = 13_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// let ct = cks.encrypt_crt(msg, basis);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt_crt(&self, message: u64, base_vec: Vec<u64>) -> Ciphertext {
        let mut ctxt_vect = Vec::with_capacity(base_vec.len());

        // Put each decomposition into a new ciphertext
        for modulus in base_vec.iter() {
            // encryption
            let ct = self
                .key
                .encrypt_with_message_modulus(message, MessageModulus(*modulus as usize));

            ctxt_vect.push(ct);
        }

        Ciphertext {
            ct_vec: ctxt_vect,
            message_modulus_vec: base_vec,
            key_id_vec: vec![],
        }
    }

    /// Decrypts a ciphertext encrypting an integer message + carries using the client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::ciphertext::Ciphertext;
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
    ///
    /// // Generate the client key and the server key:
    /// let mut cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2, 3);
    ///
    /// let msg = 27_u64;
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // Encryption of one message:
    /// let mut ct = cks.encrypt_crt(msg, basis);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt_crt(&self, ctxt: &Ciphertext) -> u64 {
        let mut val: Vec<u64> = vec![];

        // Decrypting each block individually
        for (c_i, b_i) in ctxt.ct_vec.iter().zip(ctxt.message_modulus_vec.iter()) {
            // decrypt the component i of the integer and multiply it by the radix product
            val.push(self.key.decrypt_message_and_carry(c_i) % b_i);
        }

        // Computing the inverse CRT to recompose the message
        let result = i_crt(&ctxt.message_modulus_vec, &val);

        let mut product = 1_u64;
        for b_i in ctxt.message_modulus_vec.iter() {
            product = product.wrapping_mul(*b_i);
        }

        result % product
    }

    /// Encrypts a small integer message using the client key and some moduli without padding bit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    ///
    /// let cks = ClientKey::new(PARAM_MESSAGE_3_CARRY_3, 3);
    ///
    /// let msg = 13_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// let ct = cks.encrypt_crt_not_power_of_two(msg, basis);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt_not_power_of_two(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt_crt_not_power_of_two(&self, message: u64, base_vec: Vec<u64>) -> Ciphertext {
        //Empty vector of ciphertexts
        let mut ct_vec = Vec::with_capacity(base_vec.len());

        //Put each decomposition into a new ciphertext
        for modulus in base_vec.iter() {
            // encryption
            let ct = self
                .key
                .encrypt_with_message_modulus_not_power_of_two(message, *modulus as u8);

            //put it in the vector of ciphertexts
            ct_vec.push(ct);
        }

        Ciphertext {
            ct_vec,
            message_modulus_vec: base_vec,
            key_id_vec: vec![],
        }
    }

    /// Decrypts a ciphertext encrypting an integer message with some moduli basis without
    /// padding bit.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::ciphertext::Ciphertext;
    /// use concrete_integer::client_key::ClientKey;
    /// use concrete_shortint::parameters::PARAM_MESSAGE_3_CARRY_3;
    ///
    /// // Generate the client key and the server key:
    /// let mut cks = ClientKey::new(PARAM_MESSAGE_3_CARRY_3, 3);
    ///
    /// let msg = 27_u64;
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // Encryption of one message:
    /// let mut ct = cks.encrypt_crt(msg, basis);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt_crt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt_crt_not_power_of_two(&self, ct: &Ciphertext) -> u64 {
        let mut val: Vec<u64> = vec![];

        //Decrypting each block individually
        for (c_i, b_i) in ct.ct_vec.iter().zip(ct.message_modulus_vec.iter()) {
            //decrypt the component i of the integer and multiply it by the radix product
            val.push(
                self.key
                    .decrypt_message_and_carry_not_power_of_two(c_i, *b_i as u8),
            );
        }

        //Computing the inverse CRT to recompose the message
        let result = i_crt(&ct.message_modulus_vec, &val);

        let mut product = 1_u64;
        for b_i in ct.message_modulus_vec.iter() {
            product = product.wrapping_mul(*b_i);
        }
        result % product
    }
}
