use crate::math::tensor::{AsMutTensor, Tensor};

use super::*;

/// A distribution type representing uniform sampling for boolean type.
pub struct UniformBoolean;

impl RandomGenerable<UniformBoolean> for bool {
    #[allow(unused)]
    fn sample(distribution: UniformBoolean) -> Self {
        use concrete_csprng::RandomGenerator;
        let mut gen = RandomGenerator::new(None, None);
        gen.generate_next() & 1 == 1
    }
}

/// Generates a random uniform boolean value.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::random::random_uniform_boolean;
/// let random: bool = random_uniform_boolean();
/// ```
pub fn random_uniform_boolean<T: RandomGenerable<UniformBoolean>>() -> T {
    T::sample(UniformBoolean)
}

/// Fills an `AsMutTensor` value with random boolean values.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_uniform_boolean;
/// let mut tensor = Tensor::allocate(false, 100);
/// fill_with_random_uniform_boolean(&mut tensor);
/// ```
pub fn fill_with_random_uniform_boolean<Scalar, Tensorable>(output: &mut Tensorable)
where
    Scalar: RandomGenerable<UniformBoolean>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output.as_mut_tensor().iter_mut().for_each(|s| {
        *s = random_uniform_boolean::<Scalar>();
    });
}

/// Generates a tensor of random boolean values of a given size.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::random_uniform_boolean_tensor;
/// let t: Tensor<Vec<bool>> = random_uniform_boolean_tensor(10);
/// assert_eq!(t.len(), 10);
/// ```
pub fn random_uniform_boolean_tensor<T: RandomGenerable<UniformBoolean>>(
    size: usize,
) -> Tensor<Vec<T>> {
    (0..size).map(|_| random_uniform_boolean::<T>()).collect()
}
