use crate::ciphertext::KeyId;
use crate::client_key::utils::i_crt;
use crate::CrtMultiCiphertext;

use concrete_shortint::parameters::MessageModulus;

use serde::{Deserialize, Serialize};

/// Generates a KeyId vector
///
/// Key Ids are used to choose a specific key for each block.
pub fn gen_key_id(id: &[usize]) -> Vec<KeyId> {
    id.iter().copied().map(KeyId).collect()
}

/// Client key for CRT decomposition.
///
/// As opposed to [crate::ClientKey] and [crate::CrtClientKey],
/// not all blocks in the ciphertext will use the same parameters.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct CrtMultiClientKey {
    pub(crate) keys: Vec<concrete_shortint::ClientKey>,
    pub(crate) key_ids: Vec<KeyId>,
}

/// Creates a CrtMultiClientKey from a vector of shortint keys.
///
/// Each key will encrypt one block (in order).
impl From<Vec<concrete_shortint::ClientKey>> for CrtMultiClientKey {
    fn from(keys: Vec<concrete_shortint::ClientKey>) -> Self {
        let key_ids = (0..keys.len()).map(KeyId).collect();

        Self { keys, key_ids }
    }
}

impl CrtMultiClientKey {
    /// Create a client key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::CrtMultiClientKey;
    /// use concrete_shortint::parameters::{
    ///     DEFAULT_PARAMETERS, PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3,
    /// };
    ///
    /// // Generate the client key:
    /// let cks = CrtMultiClientKey::new_many_keys(&[PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    /// ```
    pub fn new_many_keys(
        parameter_set: &[concrete_shortint::parameters::Parameters],
    ) -> CrtMultiClientKey {
        let mut key = Vec::with_capacity(parameter_set.len());
        let mut id = Vec::with_capacity(parameter_set.len());

        for (i, param) in parameter_set.iter().enumerate() {
            key.push(concrete_shortint::ClientKey::new(*param));
            id.push(KeyId(i));
        }
        CrtMultiClientKey {
            keys: key,
            key_ids: id,
        }
    }

    /// Encrypts an integer using the CRT decomposition, where each block is associated to
    /// dedicated key.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::{gen_key_id, CrtMultiClientKey};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3};
    ///
    /// // Generate the client key:
    /// let cks = CrtMultiClientKey::new_many_keys(&[PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    ///
    /// let msg = 15_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // The first two blocks are encrypted using the first key,
    /// // the third block with the second key
    /// let keys_id = gen_key_id(&[0, 0, 1]);
    /// let ct = cks.encrypt(&msg, &basis, &keys_id);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn encrypt(
        &self,
        message: &u64,
        base_vec: &[u64],
        keys_id: &[KeyId],
    ) -> CrtMultiCiphertext {
        // Empty vector of ciphertexts
        let mut ctxt_vect: Vec<concrete_shortint::Ciphertext> = Vec::new();
        let bv = base_vec.to_vec();
        let keys = keys_id.to_vec();

        // Put each decomposition into a new ciphertext
        for (modulus, id) in base_vec.iter().zip(keys_id.iter()) {
            // encryption
            let ct = self.keys[id.0]
                .encrypt_with_message_modulus(*message, MessageModulus(*modulus as usize));

            // Put it in the vector of ciphertexts
            ctxt_vect.push(ct);
        }

        CrtMultiCiphertext {
            blocks: ctxt_vect,
            moduli: bv,
            key_ids: keys,
        }
    }

    /// Decrypts an integer in the multi-key CRT settings.
    ///
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_integer::{gen_key_id, CrtMultiClientKey};
    /// use concrete_shortint::parameters::{PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3};
    ///
    /// // Generate the client key:
    /// let cks = CrtMultiClientKey::new_many_keys(&[PARAM_MESSAGE_2_CARRY_2, PARAM_MESSAGE_3_CARRY_3]);
    ///
    /// let msg = 27_u64;
    ///
    /// // Encryption of one message:
    /// let basis: Vec<u64> = vec![2, 3, 5];
    /// // The first two blocks are encrypted using the first key,
    /// // the third block with the second key
    /// let keys_id = gen_key_id(&[0, 0, 1]);
    /// let ct = cks.encrypt(&msg, &basis, &keys_id);
    ///
    /// // Decryption:
    /// let dec = cks.decrypt(&ct);
    /// assert_eq!(msg, dec);
    /// ```
    pub fn decrypt(&self, ctxt: &CrtMultiCiphertext) -> u64 {
        let mut val: Vec<u64> = vec![];

        // Decrypting each block individually
        for ((c_i, b_i), k_i) in ctxt
            .blocks
            .iter()
            .zip(ctxt.moduli.iter())
            .zip(ctxt.key_ids.iter())
        {
            // Decrypt the component i of the integer and multiply it by the radix product
            val.push(self.keys[k_i.0].decrypt_message_and_carry(c_i) % b_i);
        }

        // Computing the inverse CRT to recompose the message
        let result = i_crt(&ctxt.moduli, &val);

        let whole_modulus: u64 = ctxt.moduli.iter().copied().product();

        result % whole_modulus
    }
}
