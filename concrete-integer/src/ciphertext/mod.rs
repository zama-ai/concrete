//! This module implements the ciphertext structures.
use concrete_shortint;
use serde::{Deserialize, Serialize};

/// Id to recognize the key used to encrypt a block.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct KeyId(pub usize);

/// Structure containing a ciphertext in radix decomposition.
#[derive(Serialize, Clone, Deserialize)]
pub struct RadixCiphertext {
    /// The blocks are stored from LSB to MSB
    pub(crate) blocks: Vec<concrete_shortint::Ciphertext>,
}

impl RadixCiphertext {
    /// Returns the slice of blocks that the ciphertext is composed of.
    pub fn blocks(&self) -> &[concrete_shortint::Ciphertext] {
        &self.blocks
    }
}

/// Structure containing a ciphertext in CRT decomposition.
///
/// For this CRT decomposition, each block is encrypted using
/// the same parameters.
#[derive(Serialize, Clone, Deserialize)]
pub struct CrtCiphertext {
    pub(crate) blocks: Vec<concrete_shortint::Ciphertext>,
    pub(crate) moduli: Vec<u64>,
}

/// Structure containing a ciphertext in CRT decomposition.
///
/// For this CRT decomposition, not all blocks
/// are encrypted using the same parameters
#[derive(Serialize, Clone, Deserialize)]
pub struct CrtMultiCiphertext {
    pub(crate) blocks: Vec<concrete_shortint::Ciphertext>,
    pub(crate) moduli: Vec<u64>,
    pub(crate) key_ids: Vec<KeyId>,
}
