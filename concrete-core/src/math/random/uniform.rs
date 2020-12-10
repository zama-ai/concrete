use crate::math::tensor::{AsMutTensor, Tensor};

use super::*;

/// A distribution type representing uniform sampling for unsigned integer types. The value is
/// uniformly sampled in `[0, 2^n[` where `n` is the size of the integer type.
pub struct Uniform;

macro_rules! implement_uniform {
    ($T:ty, $bytes:literal) => {
        impl RandomGenerable<Uniform> for $T {
            #[allow(unused)]
            fn sample(distribution: Uniform) -> Self {
                use concrete_csprng::RandomGenerator;
                let mut gen = RandomGenerator::new(None, None);
                let mut buf = [0; $bytes];
                buf.iter_mut().for_each(|a| *a = gen.generate_next());
                unsafe { *(buf.as_ptr() as *const $T) }
            }
        }
    };
}

implement_uniform!(u8, 1);
implement_uniform!(u16, 2);
implement_uniform!(u32, 4);
implement_uniform!(u64, 8);
implement_uniform!(u128, 16);
implement_uniform!(i8, 1);
implement_uniform!(i16, 1);
implement_uniform!(i32, 1);
implement_uniform!(i64, 1);
implement_uniform!(i128, 1);

/// Generates a random uniform unsigned integer.
///
/// # Example
///
/// ```rust
/// # use concrete_core::math::random::random_uniform;
/// # for _ in 1..1000{
/// let random = random_uniform::<u8>();
/// let random = random_uniform::<u16>();
/// let random = random_uniform::<u32>();
/// let random = random_uniform::<u64>();
/// let random = random_uniform::<u128>();
///
/// let random = random_uniform::<i8>();
/// let random = random_uniform::<i16>();
/// let random = random_uniform::<i32>();
/// let random = random_uniform::<i64>();
/// let random = random_uniform::<i128>();
/// # }
/// ```
pub fn random_uniform<T: RandomGenerable<Uniform>>() -> T {
    T::sample(Uniform)
}

/// Fills an `AsMutTensor` value with random uniform values.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_uniform;
/// let mut tensor = Tensor::allocate(1000. as u32, 100);
/// fill_with_random_uniform(&mut tensor);
/// ```
pub fn fill_with_random_uniform<Scalar, Tensorable>(output: &mut Tensorable)
where
    Scalar: RandomGenerable<Uniform>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output.as_mut_tensor().iter_mut().for_each(|s| {
        *s = random_uniform::<Scalar>();
    });
}

/// Generates a tensor of random uniform values of a given size.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::random_uniform_tensor;
/// let t: Tensor<Vec<u64>> = random_uniform_tensor(10);
/// assert_eq!(t.len(), 10);
/// let first_val = t.get_element(0);
/// for i in 1..10{
///     assert_ne!(first_val, t.get_element(i));
/// }
/// ```
pub fn random_uniform_tensor<T: RandomGenerable<Uniform>>(size: usize) -> Tensor<Vec<T>> {
    (0..size).map(|_| random_uniform::<T>()).collect()
}
