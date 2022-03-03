//! Converting to torus values.
//!
//! The theory behind some of the homomorphic operators of the library, uses the real torus
//! $\mathbb{T} = \mathbb{R} / \mathbb{Z}$, or the set or real numbers modulo 1 (elements of the
//! torus are in $[0,1)$). In practice, floating-point number are not well suited to performing
//! operations on the torus, and we prefer to use unsigned integer values to represent them.
//! Indeed, unsigned integer can be used to encode the decimal part of the torus element with a
//! fixed precision.
//!
//! Still, in some cases, we may need to represent an unsigned integer as a torus value in
//! floating point representation. For this reason we provide the [`IntoTorus`] and [`FromTorus`]
//! traits which allow to go back and forth between an unsigned integer representation and a
//! floating point representation.

use crate::backends::core::private::math::random::{
    Gaussian, RandomGenerable, Uniform, UniformBinary, UniformTernary,
};
use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::numeric::{CastInto, FloatingPoint, Numeric, UnsignedInteger};
use std::fmt::{Debug, Display};

/// A trait that converts a torus element in unsigned integer representation to the closest
/// torus element in floating point representation.
pub trait IntoTorus<F>: Sized
where
    F: FloatingPoint,
    Self: UnsignedInteger,
{
    /// Consumes `self` and returns its closest floating point representation.
    fn into_torus(self) -> F;
}

/// A trait that converts a torus element in floating point representation into the closest torus
/// element in unsigned integer representation.
pub trait FromTorus<F>: Sized
where
    F: FloatingPoint,
    Self: UnsignedInteger,
{
    /// Consumes `input` and returns its closest unsigned integer representation.
    fn from_torus(input: F) -> Self;
}

macro_rules! implement {
    ($Type: tt) => {
        impl<F> IntoTorus<F> for $Type
        where
            F: FloatingPoint + CastInto<Self>,
            Self: CastInto<F>,
        {
            fn into_torus(self) -> F {
                let self_f: F = self.cast_into();
                return self_f * (F::TWO.powi(-(<Self as Numeric>::BITS as i32)));
            }
        }
        impl<F> FromTorus<F> for $Type
        where
            F: FloatingPoint + CastInto<Self>,
            Self: CastInto<F>,
        {
            fn from_torus(input: F) -> Self {
                let mut fract = input - F::floor(input);
                fract *= F::TWO.powi(<Self as Numeric>::BITS as i32);
                let carry = fract - F::floor(fract);
                let zero_point_five = F::ONE / F::TWO;
                if carry >= zero_point_five {
                    fract += F::ONE;
                };
                return fract.cast_into();
            }
        }
    };
}

implement!(u8);
implement!(u16);
implement!(u32);
implement!(u64);
implement!(u128);

/// A marker trait for unsigned integer types that can be used in ciphertexts, keys etc.
pub trait UnsignedTorus:
    UnsignedInteger
    + FromTorus<f64>
    + IntoTorus<f64>
    + RandomGenerable<Gaussian<f64>>
    + RandomGenerable<UniformBinary>
    + RandomGenerable<UniformTernary>
    + RandomGenerable<Uniform>
    + Display
    + Debug
{
    /// The log standard deviation used to sample gaussian keys in this precision.
    const GAUSSIAN_KEY_LOG_STD: LogStandardDev;
}

impl UnsignedTorus for u32 {
    const GAUSSIAN_KEY_LOG_STD: LogStandardDev = LogStandardDev(-30.32192809488736);
}

impl UnsignedTorus for u64 {
    const GAUSSIAN_KEY_LOG_STD: LogStandardDev = LogStandardDev(-62.32192809488736);
}
