use super::*;
use crate::math::tensor::{AsMutTensor, Tensor};

/// A distribution type representing random sampling for unsigned integer types, where the `n`
/// most significant bits are sampled randomly in `[0, 2^n[`.
pub struct UniformMsb {
    /// The number of most significant bits that must be randomly set.
    pub n: usize,
}

macro_rules! implement_uniform_some_msb {
    ($T:ty) => {
        impl RandomGenerable<UniformMsb> for $T {
            fn sample(UniformMsb { n }: UniformMsb) -> Self {
                random_uniform::<$T>() << (<$T as Numeric>::BITS - n)
            }
        }
    };
}

implement_uniform_some_msb!(u8);
implement_uniform_some_msb!(u16);
implement_uniform_some_msb!(u32);
implement_uniform_some_msb!(u64);
implement_uniform_some_msb!(u128);

/// Generates an unsigned integer whose n most significant bits are uniformly random, and the other
/// bits are zero.
///
/// # Example
///
/// ```rust
/// # use concrete_core::math::random::random_uniform_n_msb;
/// # for _ in 1..1000{
/// let random: u8 = random_uniform_n_msb(3);
/// assert!(random == 0 || random >= 32);
/// # }
/// ```
pub fn random_uniform_n_msb<T: RandomGenerable<UniformMsb>>(n: usize) -> T {
    T::sample(UniformMsb { n })
}

/// Fills an `AsMutTensor` value with values whose n msbs are random.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_uniform_n_msb;
/// let mut tensor = Tensor::allocate(8 as u8, 100);
/// fill_with_random_uniform_n_msb(&mut tensor, 5);
/// ```
pub fn fill_with_random_uniform_n_msb<Scalar, Tensorable>(output: &mut Tensorable, n: usize)
where
    Scalar: RandomGenerable<UniformMsb>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output.as_mut_tensor().iter_mut().for_each(|s| {
        *s = random_uniform_n_msb::<Scalar>(n);
    });
}

/// Generates a tensor of random uniform values, whose n msbs are sampled uniformly.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::random_uniform_n_msb_tensor;
/// let t: Tensor<Vec<u64>> = random_uniform_n_msb_tensor(10, 55);
/// assert_eq!(t.len(), 10);
/// let first_val = t.get_element(0);
/// for i in 1..10{
///     assert_ne!(first_val, t.get_element(i));
/// }
/// ```
pub fn random_uniform_n_msb_tensor<T: RandomGenerable<UniformMsb>>(
    size: usize,
    n: usize,
) -> Tensor<Vec<T>> {
    (0..size).map(|_| random_uniform_n_msb::<T>(n)).collect()
}
