//! Low-overhead homomorphic primitives.
//!
//! This module implements low-overhead fully homomorphic operations.

use std::fmt::Debug;

use serde::{Deserialize, Serialize};

pub mod bootstrap;
pub mod encoding;
pub mod ggsw;
pub mod glwe;
pub mod lwe;
pub mod secret;

/// The number plaintexts in a plaintext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
// Todo: Naming
pub struct PlaintextCount(pub usize);

/// The number messages in a messages list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
// Todo: Naming
pub struct CleartextCount(pub usize);

/// The number of ciphertexts in a ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
// Todo: Naming
pub struct CiphertextCount(pub usize);

/// The number of scalar in an LWE mask + 1 .
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
// Todo: Naming
pub struct LweSize(pub usize);

impl LweSize {
    /// Returns the associated [`LweDimension`].
    // Todo: Naming
    pub fn to_lwe_dimension(&self) -> LweDimension {
        LweDimension(self.0 - 1)
    }
}

/// The number of scalar in an LWE mask, or the length of an LWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
// Todo: Naming
pub struct LweDimension(pub usize);

impl LweDimension {
    /// Returns the associated [`LweSize`].
    // Todo: Naming
    pub fn to_lwe_size(&self) -> LweSize {
        LweSize(self.0 + 1)
    }
}

/// The number of polynomials of an GLWE mask + 1.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
// Todo: Naming
pub struct GlweSize(pub usize);

impl GlweSize {
    /// Returns the associated [`GlweDimension`].
    // Todo: Naming
    pub fn to_glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.0 - 1)
    }
}

/// The number of polynomials of an GLWE mask, or the size of an GLWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
// Todo: Naming
pub struct GlweDimension(pub usize);

impl GlweDimension {
    /// Returns the associated [`GlweSize`].
    // Todo: Naming
    pub fn to_glwe_size(&self) -> GlweSize {
        GlweSize(self.0 + 1)
    }
}
