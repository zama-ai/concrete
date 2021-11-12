//! A module containing the core backend implementation.
//!
//! This module contains a single threaded CPU implementation of the concrete scheme, which is
//! strongly biased towards x86_64 platforms. In particular, it uses fftw to perform polynomials
//! multiplication, and uses special aesni and rdseed instructions for faster random number
//! generation.

#[doc(hidden)]
pub mod private;

mod implementation;

pub use implementation::{engines, entities};
