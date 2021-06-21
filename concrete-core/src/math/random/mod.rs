//! A module containing random sampling functions.
//!
//! This module contains generic functions to sample numeric values randomly
//! according to a given distribution, for instance:
//!
//! + [`random_uniform`] samples a random unsigned integer with uniform
//! probability over the set of representable values.
//! + [`random_gaussian`] samples a random float with using a gaussian
//! distribution.
//!
//! The implementation relies on the [`RandomGenerable`] trait, which gives a
//! type the ability to be randomly generated according to a given distribution.
//! The module contains multiple implementations of this trait, for different
//! distributions. Note, though, that instead of using the [`RandomGenerable`]
//! method, you should use the provided generic functions instead:
//!
//! + [`random_uniform`]
//! + [`random_uniform_n_msb`]
//! + [`random_uniform_n_lsb`]
//! + [`random_gaussian`]
use concrete_commons::{FloatingPoint, Numeric};

use crate::math::tensor::{AsMutTensor, Tensor};

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

mod generator;
pub use generator::*;

mod secret_generator;
pub use secret_generator::*;

pub trait RandomGenerable<D: Distribution>
where
    Self: Sized,
{
    fn generate_one(generator: &mut RandomGenerator, distribution: D) -> Self;
    fn generate_tensor(
        generator: &mut RandomGenerator,
        distribution: D,
        size: usize,
    ) -> Tensor<Vec<Self>> {
        (0..size)
            .map(|_| Self::generate_one(generator, distribution))
            .collect()
    }
    fn fill_tensor<Tens>(generator: &mut RandomGenerator, distribution: D, tensor: &mut Tens)
    where
        Tens: AsMutTensor<Element = Self>,
    {
        tensor.as_mut_tensor().iter_mut().for_each(|s| {
            *s = Self::generate_one(generator, distribution);
        });
    }
}

/// A marker trait for types representing distributions.
pub trait Distribution: seal::Sealed + Copy {}
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
