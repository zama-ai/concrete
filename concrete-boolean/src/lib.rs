//! Welcome the the `concrete-boolean` documentation!
//!
//! # Description
//! This library makes it possible to execute boolean gates over encrypted bits.
//! It allows to execute a boolean circuit on an untrusted server because both circuit inputs and
//! outputs are kept private.
//! Data are indeed encrypted on the client side, before being sent to the server.
//! On the server side every computation is performed on ciphertexts.
//! The server however has to know the boolean circuit to be evaluated.
//! At the end of the computation, the server returns the encryption of the result to the user.
//!
//!
//!
//! # Quick Example
//!
//! The following piece of code shows how to generate keys and run a small Boolean circuit
//! homomorphically.
//!
//! ```rust
//! extern crate concrete_boolean;
//! use concrete_boolean::gen_keys;
//!
//! // We generate a set of client/server keys, using the default parameters:
//! let (client_key, server_key) = gen_keys();
//!
//! // We use the client secret key to encrypt two messages:
//! let ct_1 = client_key.encrypt(true);
//! let ct_2 = client_key.encrypt(false);
//!
//! // We use the server public key to execute a boolean circuit:
//! // if ((NOT ct_2) NAND (ct_1 AND ct_2)) then (NOT ct_2) else (ct_1 AND ct_2)
//! let ct_3 = server_key.not(&ct_2);
//! let ct_4 = server_key.and(&ct_1, &ct_2);
//! let ct_5 = server_key.nand(&ct_3, &ct_4);
//! let ct_6 = server_key.mux(&ct_5, &ct_3, &ct_4);
//!
//! // We use the client key to decrypt the output of the circuit:
//! let output = client_key.decrypt(&ct_6);
//! assert_eq!(output, true);
//! ```

use crate::client_key::ClientKey;
use crate::parameters::DEFAULT_PARAMETERS;
use crate::server_key::ServerKey;

pub mod ciphertext;
pub mod client_key;
pub mod parameters;
pub mod server_key;

/// The scaling factor used for the plaintext
pub(crate) const PLAINTEXT_LOG_SCALING_FACTOR: usize = 3;

/// The plaintext associated with true: 1/8
pub(crate) const PLAINTEXT_TRUE: u32 = 1 << (32 - PLAINTEXT_LOG_SCALING_FACTOR);

/// The plaintext associated with false: -1/8
pub(crate) const PLAINTEXT_FALSE: u32 = 7 << (32 - PLAINTEXT_LOG_SCALING_FACTOR);

/// tool to generate random booleans
#[cfg(test)]
pub(crate) fn random_boolean() -> bool {
    // create a random generator
    let mut generator = concrete_core::math::random::RandomGenerator::new(None);

    // generate a bit
    let n: u32 = generator.random_uniform_binary();

    // convert it to boolean and return
    n != 0
}

/// tool to generate random integers
#[cfg(test)]
pub(crate) fn random_integer() -> u32 {
    // create a random generator
    let mut generator = concrete_core::math::random::RandomGenerator::new(None);

    // generate a bit
    generator.random_uniform()
}

/// Generate a couple of client and server keys with the default cryptographic parameters:
/// `DEFAULT_PARAMETERS`.
/// The client is the one generating both keys.
/// * the client key is used to encrypt and decrypt and has to be kept secret;
/// * the server key is used to perform homomorphic operations on the server side and it is
/// meant to be published (the client sends it to the server).
///
/// ```rust
/// use concrete_boolean::gen_keys;
/// // generate the client key and the server key:
/// let (cks, sks) = gen_keys();
/// ```
pub fn gen_keys() -> (ClientKey, ServerKey) {
    // generate the client key
    let cks = ClientKey::new(&DEFAULT_PARAMETERS);

    // generate the server key
    let sks = ServerKey::new(&cks);

    // return
    (cks, sks)
}
