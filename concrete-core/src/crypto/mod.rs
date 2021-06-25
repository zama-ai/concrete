//! Low-overhead homomorphic primitives.
//!
//! This module implements low-overhead fully homomorphic operations.

use std::fmt::Debug;

pub mod bootstrap;
pub mod encoding;
pub mod ggsw;
pub mod glwe;
pub mod lwe;
pub mod secret;
