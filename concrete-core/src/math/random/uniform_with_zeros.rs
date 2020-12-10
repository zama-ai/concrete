use super::*;
use crate::math::tensor::{AsMutTensor, Tensor};

/// A distribution type taht samples a uniform value with probability `1 - prob_zero`, and a zero
/// value with probaibility `prob_zero`.
pub struct UniformWithZeros {
    /// The probability of the output being a zero
    pub prob_zero: f32,
}

#[allow(unused_macros)]
macro_rules! implement_uniform_with_zeros {
    ($T:ty, $bits:literal) => {
        impl RandomGenerable<UniformWithZeros> for $T {
            #[allow(unused)]
            fn sample(UniformWithZeros { prob_zero }: UniformWithZeros) -> Self {
                let float_sample = random_uniform::<u32>() as f32 / u32::MAX as f32;
                if float_sample < prob_zero {
                    <$T>::ZERO
                } else {
                    random_uniform::<$T>()
                }
            }
        }
    };
}

implement_uniform_with_zeros!(u8, 1);
implement_uniform_with_zeros!(u16, 2);
implement_uniform_with_zeros!(u32, 4);
implement_uniform_with_zeros!(u64, 8);
implement_uniform_with_zeros!(u128, 16);
implement_uniform_with_zeros!(i8, 1);
implement_uniform_with_zeros!(i16, 2);
implement_uniform_with_zeros!(i32, 4);
implement_uniform_with_zeros!(i64, 8);
implement_uniform_with_zeros!(i128, 16);

/// Generates a random uniform unsigned integer with probability `1-prob_zero`, and a zero value
/// with probability `prob_zero`.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::random::random_uniform_with_zeros;
/// # for _ in 1..1000{
/// let random = random_uniform_with_zeros::<u8>(0.5);
/// let random = random_uniform_with_zeros::<u16>(0.5);
/// let random = random_uniform_with_zeros::<u32>(0.5);
/// let random = random_uniform_with_zeros::<u64>(0.5);
/// let random = random_uniform_with_zeros::<u128>(0.5);
/// assert_eq!(random_uniform_with_zeros::<u128>(1.), 0);
/// assert_ne!(random_uniform_with_zeros::<u128>(0.), 0);
/// # }
/// ```
pub fn random_uniform_with_zeros<T: RandomGenerable<UniformWithZeros>>(prob_zero: f32) -> T {
    T::sample(UniformWithZeros { prob_zero })
}

/// Fills an `AsMutTensor` value with random boolean values.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_uniform_with_zeros;
/// let mut tensor = Tensor::allocate(10 as u8, 100);
/// fill_with_random_uniform_with_zeros(&mut tensor, 0.5);
/// ```
pub fn fill_with_random_uniform_with_zeros<Scalar, Tensorable>(
    output: &mut Tensorable,
    prob_zero: f32,
) where
    Scalar: RandomGenerable<UniformWithZeros>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output.as_mut_tensor().iter_mut().for_each(|s| {
        *s = random_uniform_with_zeros::<Scalar>(prob_zero);
    });
}

/// Generates a tensor of a given size, whose coefficients are random uniform with probability
/// `1-prob_zero`, and zero with probability `prob_zero`.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::random_uniform_with_zeros_tensor;
/// let t: Tensor<Vec<u64>> = random_uniform_with_zeros_tensor(10, 0.);
/// assert_eq!(t.len(), 10);
/// t.iter().for_each(|a| assert_ne!(*a, 0));
/// let t: Tensor<Vec<u64>> = random_uniform_with_zeros_tensor(10, 1.);
/// t.iter().for_each(|a| assert_eq!(*a, 0));
/// ```
pub fn random_uniform_with_zeros_tensor<T: RandomGenerable<UniformWithZeros>>(
    size: usize,
    prob_zero: f32,
) -> Tensor<Vec<T>> {
    (0..size)
        .map(|_| random_uniform_with_zeros::<T>(prob_zero))
        .collect()
}
