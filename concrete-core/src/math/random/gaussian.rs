use std::f64::consts::SQRT_2;

use concrete_commons::numeric::{CastInto, Numeric};

use crate::math::torus::{FromTorus, UnsignedTorus};

use statrs::function::erf::erf_inv;

use super::*;

/// A distribution type representing random sampling of floating point numbers, following a
/// gaussian distribution.
#[derive(Clone, Copy)]
pub struct Gaussian<T: FloatingPoint> {
    /// The standard deviation of the distribution.
    pub std: T,
    /// The mean of the distribution.
    pub mean: T,
}

macro_rules! implement_gaussian {
    ($T:ty, $S:ty) => {
        impl RandomGenerable<Gaussian<$T>> for $T {
            fn generate_one(
                generator: &mut RandomGenerator,
                Gaussian { std, mean }: Gaussian<$T>,
            ) -> Self {
                let mut uniform_rand = 0 as $S;
                for _ in 0..<$S as Numeric>::BITS / 8 {
                    uniform_rand <<= 8;
                    uniform_rand += generator.generate_next() as $S;
                }
                let size = <$T>::BITS as i32;
                let mut u: $T = uniform_rand.cast_into();
                u *= <$T>::TWO.powi(-size + 1);
                erf_inv(u.into()) as $T * std * SQRT_2 as $T + mean
            }
        }
    };
}

implement_gaussian!(f32, i32);
implement_gaussian!(f64, i64);

impl<Torus> RandomGenerable<Gaussian<f64>> for Torus
where
    Torus: UnsignedTorus,
{
    fn generate_one(generator: &mut RandomGenerator, distribution: Gaussian<f64>) -> Self {
        <Torus as FromTorus<f64>>::from_torus(f64::generate_one(generator, distribution))
    }
}
