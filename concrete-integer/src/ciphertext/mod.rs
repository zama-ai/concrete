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



pub trait IntegerCiphertext: Clone{
    fn from_blocks(blocks: Vec<concrete_shortint::Ciphertext>) -> Self;
    fn blocks(&self) -> &[concrete_shortint::Ciphertext];
    fn blocks_mut(&mut self) -> &mut [concrete_shortint::Ciphertext];
    fn moduli(&self) -> Vec<u64>{
        self.blocks().iter().map(|x| x.message_modulus.0 as u64).collect()
    }
}

impl IntegerCiphertext for RadixCiphertext{
    fn blocks(&self) -> &[concrete_shortint::Ciphertext] {
        &self.blocks
    }
    fn blocks_mut(&mut self) -> &mut [concrete_shortint::Ciphertext] {
        &mut self.blocks
    }
    fn from_blocks(blocks: Vec<concrete_shortint::Ciphertext>) -> Self{
        Self{blocks}
    }

}

impl IntegerCiphertext for CrtCiphertext{
    fn blocks(&self) -> &[concrete_shortint::Ciphertext] {
        &self.blocks
    }
    fn blocks_mut(&mut self) -> &mut [concrete_shortint::Ciphertext] {
        &mut self.blocks
    }
    fn from_blocks(blocks: Vec<concrete_shortint::Ciphertext>) -> Self{
        let moduli = blocks.iter().map(|x| x.message_modulus.0 as u64).collect();
        Self{blocks, moduli}
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
