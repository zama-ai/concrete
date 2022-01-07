//! A module containing the Optalysys backend implementation.
//!
//! This module contains the implementation of some functions of the concrete scheme accelerated
//! with the Optalysys hardware.

#[doc(hidden)]
pub mod private;

pub(crate) mod implementation;

pub use implementation::{engines, entities};
