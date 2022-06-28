#![allow(clippy::excessive_precision)]
//! Welcome the the `concrete-integer` documentation!
//!
//! # Description
//!
//! This library makes it possible to execute modular operations over encrypted integer.
//!
//! It allows to execute an integer circuit on an untrusted server because both circuit inputs
//! outputs are kept private.
//!
//! Data are encrypted on the client side, before being sent to the server.
//! On the server side every computation is performed on ciphertexts
//!
//! # Quick Example
//!
//! The following piece of code shows how to generate keys and run a integer circuit
//! homomorphically.
//!
//! ```rust
//! use concrete_integer::gen_keys;
//! use concrete_shortint::parameters::PARAM_MESSAGE_2_CARRY_2;
//!
//! //4 blocks for the radix decomposition
//! let number_of_blocks = 4;
//! // Modulus = (2^2)*4 = 2^8 (from the parameters chosen and the number of blocks
//! let modulus = 1 << 8;
//!
//! // Generation of the client/server keys, using the default parameters:
//! let (mut client_key, mut server_key) = gen_keys(&PARAM_MESSAGE_2_CARRY_2, number_of_blocks);
//!
//! let msg1 = 153;
//! let msg2 = 125;
//!
//! // Encryption of two messages using the client key:
//! let ct_1 = client_key.encrypt(msg1);
//! let ct_2 = client_key.encrypt(msg2);
//!
//! // Homomorphic evaluation of an integer circuit (here, an addition) using the server key:
//! let ct_3 = server_key.unchecked_add(&ct_1, &ct_2);
//!
//! // Decryption of the ciphertext using the client key:
//! let output = client_key.decrypt(&ct_3);
//! assert_eq!(output, (msg1 + msg2) % modulus);
//! ```
//!
//! # Warning
//! This uses cryptographic parameters from the `concrete-shortint` crates.
//! Currently, the radix approach is only compatible with parameter sets such
//! that the message and carry buffers have the same size.
extern crate core;

pub mod ciphertext;
pub mod client_key;
pub mod crt;
#[cfg(any(test, feature = "internal-keycache"))]
pub mod keycache;
pub mod parameters;
pub mod server_key;
#[cfg(doctest)]
mod test_user_docs;
pub mod treepbs;
pub mod wopbs;

pub use ciphertext::Ciphertext;
pub use client_key::ClientKey;
pub use server_key::{CheckError, ServerKey};

/// Generate a couple of client and server keys with given parameters
///
/// * the client key is used to encrypt and decrypt and has to be kept secret;
/// * the server key is used to perform homomorphic operations on the server side and it is meant to
///   be published (the client sends it to the server).
///
/// ```rust
/// use concrete_integer::gen_keys;
/// use concrete_shortint::parameters::DEFAULT_PARAMETERS;
///
/// let size = 4;
///
/// // generate the client key and the server key:
/// let (cks, sks) = gen_keys(&DEFAULT_PARAMETERS, size);
/// ```
pub fn gen_keys(
    parameters_set: &concrete_shortint::parameters::Parameters,
    size: usize,
) -> (ClientKey, ServerKey) {
    let cks = ClientKey::new(*parameters_set, size);
    let sks = ServerKey::new(&cks);

    (cks, sks)
}
