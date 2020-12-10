use super::*;
use crate::math::tensor::{AsMutTensor, Tensor};

/// A distribution type representing random sampling for unsigned integer type, where the `n`
/// least significant bits are sampled in `[0, 2^n[`.
pub struct UniformLsb {
    /// The number of least significant bits that should be set randomly.
    pub n: usize,
}

macro_rules! implement_uniform_some_lsb {
    ($T:ty) => {
        impl RandomGenerable<UniformLsb> for $T {
            fn sample(UniformLsb { n }: UniformLsb) -> Self {
                random_uniform::<$T>() >> (<$T as Numeric>::BITS - n)
            }
        }
    };
}

implement_uniform_some_lsb!(u8);
implement_uniform_some_lsb!(u16);
implement_uniform_some_lsb!(u32);
implement_uniform_some_lsb!(u64);
implement_uniform_some_lsb!(u128);

/// Generates an unsigned integer whose n least significant bits are uniformly random, and the other
/// bits are zero.
///
/// # Example
///
/// ```rust
/// # use concrete_core::math::random::random_uniform_n_lsb;
/// # for _ in 1..1000{
/// let random: u8 = random_uniform_n_lsb(3);
/// assert!(random <= 7 as u8);
/// # }
/// ```
pub fn random_uniform_n_lsb<T: RandomGenerable<UniformLsb>>(n: usize) -> T {
    T::sample(UniformLsb { n })
}

/// Fills an `AsMutTensor` value with random values whose n lsbs are sampled uniformly.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_uniform_n_lsb;
/// let mut tensor = Tensor::allocate(0 as u8, 100);
/// fill_with_random_uniform_n_lsb(&mut tensor, 3);
/// ```
pub fn fill_with_random_uniform_n_lsb<Scalar, Tensorable>(output: &mut Tensorable, n: usize)
where
    Scalar: RandomGenerable<UniformLsb>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output.as_mut_tensor().iter_mut().for_each(|s| {
        *s = random_uniform_n_lsb::<Scalar>(n);
    });
}

/// Generates a tensor of random uniform values, whose n lsbs are sampled uniformly.
///
/// # Example
///
/// ```rust
/// # use concrete_core::math::tensor::Tensor;
/// # use concrete_core::math::random::random_uniform_n_lsb_tensor;
/// let t: Tensor<Vec<u64>> = random_uniform_n_lsb_tensor(10, 55);
/// assert_eq!(t.len(), 10);
/// let first_val = t.get_element(0);
/// for i in 1..10{
///     assert_ne!(first_val, t.get_element(i));
/// }
/// ```
pub fn random_uniform_n_lsb_tensor<T: RandomGenerable<UniformLsb>>(
    size: usize,
    n: usize,
) -> Tensor<Vec<T>> {
    (0..size).map(|_| random_uniform_n_lsb::<T>(n)).collect()
}
