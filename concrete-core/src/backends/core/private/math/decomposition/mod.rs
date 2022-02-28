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
use std::fmt::Debug;

#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::prelude::numeric::SignedInteger;
pub use decomposer::*;
pub use iter::*;
pub use term::*;

mod decomposer;
mod iter;
mod term;
#[cfg(test)]
mod tests;

/// The level of a given term of a decomposition.
///
/// When decomposing an integer over the $l$ levels, this type represent the level (in $[0,l)$)
/// currently manipulated.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct DecompositionLevel(pub usize);

pub(crate) fn torus_small_sign_decompose<Scalar>(res: &mut [Scalar], val: Scalar, base_log: usize)
where
    Scalar: UnsignedTorus,
    Scalar::Signed: SignedInteger,
{
    let mut tmp: Scalar;
    let mut carry = Scalar::ZERO;
    let mut previous_carry: Scalar;
    let block_bit_mask: Scalar = (Scalar::ONE << base_log) - Scalar::ONE;
    let msb_block_mask: Scalar = Scalar::ONE << (base_log - 1);

    // compute the decomposition from LSB to MSB (because of the carry)
    for i in (0..res.len()).rev() {
        previous_carry = carry;
        tmp = (val >> (Scalar::BITS - base_log * (i + 1))) & block_bit_mask;
        carry = tmp & msb_block_mask;
        tmp = tmp.wrapping_add(previous_carry);
        carry |= tmp & msb_block_mask; // 0000...0001000 or 0000...0000000
        res[i] = ((tmp.into_signed()) - ((carry << 1).into_signed())).into_unsigned();
        carry >>= base_log - 1; // 000...0001 or 000...0000
    }
}
