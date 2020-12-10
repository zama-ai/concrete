//! A module containing random sampling functions.
//!
//! This module contains generic functions to sample numeric values randomly according to a given
//! distribution, for instance:
//!
//! + [`random_uniform`] samples a random unsigned integer with uniform probability over
//! the set of representable values.
//! + [`random_gaussian`] samples a random float with using a gaussian distribution.
//!
//! The implementation relies on the [`RandomGenerable`] trait, which gives a type the ability to
//! be randomly generated according to a given distribution. The module contains multiple
//! implementations of this trait, for different distributions. Note, though, that instead of
//! using the [`RandomGenerable`] method, you should use the provided generic functions instead:
//!
//! + [`random_uniform`]
//! + [`random_uniform_n_msb`]
//! + [`random_uniform_n_lsb`]
//! + [`random_gaussian`]
use crate::numeric::{FloatingPoint, Numeric};

#[cfg(test)]
mod tests;

mod uniform;
pub use uniform::*;

mod uniform_msb;
pub use uniform_msb::*;

mod uniform_lsb;
pub use uniform_lsb::*;

mod gaussian;
pub use gaussian::*;

mod uniform_with_zeros;
pub use uniform_with_zeros::*;

mod uniform_boolean;
pub use uniform_boolean::*;

/// A trait allowing a type to be randomly generated with a distribution represented by the generic
/// `D` type.
///
/// To implement the trait, the method `sample` must be implemented. It takes a value of type
/// `D` as input. For instance, when implementing the trait with the [`Gaussian`] distribution type,
/// the parameters of the distribution can be passed to the `sample` method by using the fields
/// `std` and `mean` of `Gaussian`.
pub trait RandomGenerable<D: Distribution> {
    /// A method which allows to sample a value of type `Self` using the distribution
    /// `Distribution`.
    fn sample(distribution: D) -> Self;
}

/// A marker trait for types representing distributions.
pub trait Distribution: seal::Sealed {}
mod seal {
    pub trait Sealed {}
    impl Sealed for super::Uniform {}
    impl Sealed for super::UniformMsb {}
    impl Sealed for super::UniformLsb {}
    impl Sealed for super::UniformWithZeros {}
    impl Sealed for super::UniformBoolean {}
    impl<T: super::FloatingPoint> Sealed for super::Gaussian<T> {}
}
impl Distribution for Uniform {}
impl Distribution for UniformMsb {}
impl Distribution for UniformLsb {}
impl Distribution for UniformWithZeros {}
impl Distribution for UniformBoolean {}
impl<T: FloatingPoint> Distribution for Gaussian<T> {}
