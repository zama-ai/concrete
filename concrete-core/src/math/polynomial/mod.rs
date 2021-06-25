//! A module to manipulate polynomials.
//!
//! This module allows to manipulate modular polynomials In particular, we
//! provide three generic types to manipulate such objects:
//!
//! + [`Monomial`], which represents a free monomial term (not bound to a given
//! modular degree) + [`Polynomial`], which represents a dense polynomial of a
//! given degree. + [`PolynomialList`], which represent a set of polynomials
//! with the same degree, on which operations can be performed.

use serde::{Deserialize, Serialize};

pub use list::*;
pub use monomial::*;
pub use polynomial::*;

#[cfg(test)]
mod tests;

mod list;
mod monomial;
#[allow(clippy::module_inception)]
mod polynomial;

/// The degree of a monomial.
///
/// Assuming a monomial $aX^N$, this returns the $N$ value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonomialDegree(pub usize);

/// The number of coefficients of a polynomial.
///
/// Assuming a polynomial $a_0 + a_1X + /dots + a_nX^N$, this returns $N+1$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolynomialSize(pub usize);

/// The number of polynomials in a polynomial list.
///
/// Assuming a polynomial list, this return the number of polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolynomialCount(pub usize);
