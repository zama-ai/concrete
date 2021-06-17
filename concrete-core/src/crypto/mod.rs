//! Low-overhead homomorphic primitives.
//!
//! This module implements low-overhead fully homomorphic operations.

use std::fmt::{Debug, Display};

use concrete_commons::numeric::{CastFrom, CastInto, UnsignedInteger};

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

/// A marker trait for unsigned integer types that can be used in ciphertexts,
/// keys etc.
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
