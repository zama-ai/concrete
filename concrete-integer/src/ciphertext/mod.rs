//! This module implements the ciphertext structure containing an encryption of an integer message.
use concrete_shortint;
use serde::{Deserialize, Serialize};

/// Id to recognize the key used to encrypt a block.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct KeyId(pub usize);

/// A structure containing a ciphertext, meant to encrypt an (large) integer message.
/// It is used to evaluate a integer circuits homomorphically.
#[derive(Serialize, Clone, Deserialize)]
pub struct Ciphertext {
    pub(crate) ct_vec: Vec<concrete_shortint::ciphertext::Ciphertext>,
    pub(crate) message_modulus_vec: Vec<u64>,

    // KeyId used to identify the encryption key of a bloc
    // (mainly used in the CRT)
    pub(crate) key_id_vec: Vec<KeyId>,
}

impl Ciphertext {
    /// Returns the slice of blocks that the ciphertext is composed of.
    pub fn blocks(&self) -> &[concrete_shortint::Ciphertext] {
        &self.ct_vec
    }
}
