#![deny(rustdoc::broken_intra_doc_links)]
//! Common tools for the concrete packages
//!
//! # Dispersion
//! This module contains the functions used to compute the variance, standard
//! deviation, etc.
//!
//! # Key kinds
//! This module contains types to manage the different kinds of secret keys.
//!
//! # Parameters
//! This module contains structures that wrap unsigned integer parameters of
//! concrete, like the ciphertext dimension or the polynomial degree.
//!
//! # Numeric
//! This module contains types and traits used to handle numeric types in a
//! unified manner in concrete: it defines methods that can be used on custom
//! traits [`UnsignedInteger`](numeric::UnsignedInteger), [`SignedInteger`](numeric::SignedInteger)
//! and [`FloatingPoint`](numeric::FloatingPoint),
//! regardless of the
//! number of bits in the representation.

pub mod dispersion;
pub mod key_kinds;
pub mod markers;
pub mod numeric;
pub mod parameters;
