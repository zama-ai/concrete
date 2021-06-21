//! Low-overhead homomorphic primitives.
//!
//! This module implements low-overhead fully homomorphic operations.

use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use concrete_commons::{CastFrom, CastInto, UnsignedInteger};

use crate::math::decomposition::SignedDecomposable;
use crate::math::random::{Gaussian, RandomGenerable, Uniform};
use crate::math::torus::{FromTorus, IntoTorus};

pub mod bootstrap;
pub mod cross;
pub mod encoding;
pub mod ggsw;
pub mod glwe;
pub mod lwe;
pub mod secret;

/// A marker trait for unsigned integer types that can be used in ciphertexts, keys etc.
pub trait UnsignedTorus:
    UnsignedInteger
    + FromTorus<f64>
    + IntoTorus<f64>
    + SignedDecomposable
    + RandomGenerable<Gaussian<f64>>
    + RandomGenerable<Uniform>
    + Display
    + Debug
    + CastFrom<bool>
    + CastFrom<f64>
    + CastInto<f64>
{
}

impl UnsignedTorus for u32 {}
impl UnsignedTorus for u64 {}

/// The number plaintexts in a plaintext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct PlaintextCount(pub usize);

/// The number messages in a messages list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct CleartextCount(pub usize);

/// The number of ciphertexts in a ciphertext list.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct CiphertextCount(pub usize);

/// The number of scalar in an LWE mask + 1 .
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct LweSize(pub usize);

impl LweSize {
    /// Returns the associated [`LweDimension`].
    pub fn to_lwe_dimension(&self) -> LweDimension {
        LweDimension(self.0 - 1)
    }
}

/// The number of scalar in an LWE mask, or the length of an LWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct LweDimension(pub usize);

impl LweDimension {
    /// Returns the associated [`LweSize`].
    pub fn to_lwe_size(&self) -> LweSize {
        LweSize(self.0 + 1)
    }
}

/// The number of polynomials of an GLWE mask + 1.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Serialize, Deserialize)]
pub struct GlweSize(pub usize);

impl GlweSize {
    /// Returns the associated [`GlweDimension`].
    pub fn to_glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.0 - 1)
    }
}

/// The number of polynomials of an GLWE mask, or the size of an GLWE secret key.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct GlweDimension(pub usize);

impl GlweDimension {
    /// Returns the associated [`GlweSize`].
    pub fn to_glwe_size(&self) -> GlweSize {
        GlweSize(self.0 + 1)
    }
}
