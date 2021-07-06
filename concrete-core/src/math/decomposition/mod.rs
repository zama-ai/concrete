//! Signed decomposition of unsigned integers.
//!
//! Multiple homomorphic operations used in the concrete scheme use a signed decomposition to reduce
//! the amount of noise. This module contains a [`SignedDecomposer`] which offer a clean api for
//! this decomposition.
//!
//! # Description
//!
//! We assume a number $\theta$ lives in $\mathbb{Z}/q\mathbb{Z}$, with $q$ a power of two. Such
//! a number can also be seen as a signed integer in $[ -\frac{q}{2}; \frac{q}{2}-1]$. Assuming a
//! given base $B=2^{b}$ and a number of levels $l$ such that $B^l\leq q$, such a $\theta$ can be
//! approximately decomposed as:
//! $$
//!     \theta \approx \sum_{i=1}^l\tilde{\theta}_i\frac{q}{B^i}
//! $$
//! With the $\tilde{\theta}_i\in[-\frac{B}{2}, \frac{B}{2}-1]$. When $B^l = q$, the decomposition
//! is no longer an approximation, and becomes exact. The rationale behind using an approximate
//! decomposition like that, is that when using this decomposition the approximation error will be
//! located in the least significant bits, which are already erroneous.

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

mod decomposer;
mod iter;
mod term;
#[cfg(test)]
mod tests;

pub use decomposer::*;
pub use iter::*;
pub use term::*;

/// The logarithm of the base used in a decomposition.
///
/// When decomposing an integer over powers of the $B=2^b$ basis, this type represents the $b$
/// value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionBaseLog(pub usize);

/// The number of levels used in a decomposition.
///
/// When decomposing an integer over the $l$ levels, this type represents the $l$ value.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevelCount(pub usize);

/// The level of a given term of a decomposition.
///
/// When decomposing an integer over the $l$ levels, this type represent the level (in $[0,l)$)
/// currently manipulated.
#[derive(Debug, PartialEq, Eq, Copy, Clone, Deserialize, Serialize)]
pub struct DecompositionLevel(pub usize);
