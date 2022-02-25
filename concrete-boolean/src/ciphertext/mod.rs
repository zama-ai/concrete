//! An encryption of a boolean message.
//!
//! This module implements the ciphertext structure containing an encryption of a Boolean message.

use concrete_core::prelude::*;
use serde::{Deserialize, Serialize};

/// A structure containing a ciphertext, meant to encrypt a Boolean message.
///
/// It is used to evaluate a Boolean circuits homomorphically.
#[derive(Serialize, Clone, Deserialize)]
pub struct Ciphertext(pub(crate) LweCiphertext32);
