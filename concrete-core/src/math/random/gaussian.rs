use crate::crypto::UnsignedTorus;
use crate::math::tensor::{AsMutSlice, AsMutTensor, Tensor};
use crate::math::torus::FromTorus;
use crate::numeric::{CastInto, FloatingPoint, Numeric};

use super::*;

/// A distribution type representing random sampling of floating point numbers, following a
/// gaussian distribution.
pub struct Gaussian<T: FloatingPoint> {
    /// The standard deviation of the distribution.
    pub std: T,
    /// The mean of the distribution.
    pub mean: T,
}

macro_rules! implement_gaussian {
    ($T:ty, $S:ty) => {
        impl RandomGenerable<Gaussian<$T>> for ($T, $T) {
            fn sample(Gaussian { std, mean }: Gaussian<$T>) -> Self {
                let output: ($T, $T);
                let mut uniform_rand = vec![0 as $S; 2];
                let mut gen = concrete_csprng::RandomGenerator::new(None, None);
                loop {
                    let n_bytes = (<$S as Numeric>::BITS * 2) / 8;
                    let uniform_rand_bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            uniform_rand.as_mut_ptr() as *mut u8,
                            n_bytes,
                        )
                    };
                    uniform_rand_bytes
                        .iter_mut()
                        .for_each(|a| *a = gen.generate_next());
                    let size = <$T>::BITS as i32;
                    let mut u: $T = uniform_rand[0].cast_into();
                    u *= <$T>::TWO.powi(-size + 1);
                    let mut v: $T = uniform_rand[1].cast_into();
                    v *= <$T>::TWO.powi(-size + 1);
                    let s = u.powi(2) + v.powi(2);
                    if (s > <$T>::ZERO && s < <$T>::ONE) {
                        let cst = std * (-<$T>::TWO * s.ln() / s).sqrt();
                        output = (u * cst + mean, v * cst + mean);
                        break;
                    }
                }
                output
            }
        }
    };
}

implement_gaussian!(f32, i32);
implement_gaussian!(f64, i64);

impl<Torus> RandomGenerable<Gaussian<f64>> for (Torus, Torus)
where
    Torus: UnsignedTorus,
{
    fn sample(distribution: Gaussian<f64>) -> Self {
        let (s1, s2) = <(f64, f64)>::sample(distribution);
        (
            <Torus as FromTorus<f64>>::from_torus(s1),
            <Torus as FromTorus<f64>>::from_torus(s2),
        )
    }
}

impl<Torus> RandomGenerable<Gaussian<f64>> for Torus
where
    Torus: UnsignedTorus,
{
    fn sample(distribution: Gaussian<f64>) -> Self {
        let (s1, _) = <(f64, f64)>::sample(distribution);
        <Torus as FromTorus<f64>>::from_torus(s1)
    }
}

/// Generates two floating point values drawn from a gaussian distribution with input mean and
/// standard deviation.
///
/// # Example
///
/// ```rust
/// use concrete_core::math::random::random_gaussian;
/// # let mut n_inside_3_sigma: usize = 0;
/// # for _ in 1..1000{
/// // for f32
/// let (g1, g2): (f32, f32) = random_gaussian(0. as f32, 1. as f32);
/// // check that both samples are in 6 sigmas.
/// assert!(g1.abs() <= 6.);
/// assert!(g2.abs() <= 6.);
/// # if g1.abs() <= 3. { n_inside_3_sigma += 1 }
/// # if g2.abs() <= 3. { n_inside_3_sigma += 1 }
/// # }
/// # let inside_ratio = n_inside_3_sigma as f32 / 2000. ;
/// # assert!(inside_ratio >= 0.99,
/// #     "failed the 3 sigma test. Ratio inside 3 sigma: {}",
/// #     inside_ratio);
/// # let mut n_inside_3_sigma: usize = 0;
/// # for _ in 1..1000{
/// // for f64
/// let (g1, g2): (f64,f64) = random_gaussian(0. as f64, 1. as f64);
/// // check that both samples are in 6 sigmas.
/// assert!(g1.abs() <= 6.);
/// assert!(g2.abs() <= 6.);
/// # if g1.abs() <= 3. { n_inside_3_sigma += 1 }
/// # if g2.abs() <= 3. { n_inside_3_sigma += 1 }
/// # }
/// # let inside_ratio = n_inside_3_sigma as f32 / 2000. ;
/// # assert!(inside_ratio >= 0.99,
/// #     "failed the 3 sigma test. Ratio inside 3 sigma: {}",
/// #     inside_ratio);
/// ```
pub fn random_gaussian<Float, Scalar>(mean: Float, std: Float) -> (Scalar, Scalar)
where
    Float: FloatingPoint,
    (Scalar, Scalar): RandomGenerable<Gaussian<Float>>,
{
    <(Scalar, Scalar)>::sample(Gaussian { std, mean })
}

/// Fills an `AsMutTensor` value with random gaussian values.
///
/// # Example
///
/// ```
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::fill_with_random_gaussian;
/// let mut tensor = Tensor::allocate(1000. as f32, 100);
/// fill_with_random_gaussian(&mut tensor, 0., 1.);
/// tensor.iter().for_each(|t| assert_ne!(*t, 1000.));
/// ```
pub fn fill_with_random_gaussian<Float, Scalar, Tensorable>(
    output: &mut Tensorable,
    mean: Float,
    std: Float,
) where
    Float: FloatingPoint,
    (Scalar, Scalar): RandomGenerable<Gaussian<Float>>,
    Tensorable: AsMutTensor<Element = Scalar>,
{
    output
        .as_mut_tensor()
        .as_mut_slice()
        .chunks_mut(2)
        .for_each(|s| {
            let (g1, g2) = random_gaussian::<Float, Scalar>(mean, std);
            if let Some(elem) = s.get_mut(0) {
                *elem = g1;
            }
            if let Some(elem) = s.get_mut(1) {
                *elem = g2;
            }
        });
}

/// Generates a new tensor of floating point values, randomly sampled from a gaussian distribution:
///
/// # Example
///
/// ```rust
/// use concrete_core::math::tensor::Tensor;
/// use concrete_core::math::random::random_gaussian_tensor;
/// let tensor: Tensor<Vec<f32>> = random_gaussian_tensor(10_000, 0. as f32, 1. as f32);
/// assert_eq!(tensor.len(), 10_000);
/// tensor.iter()
///     .for_each(|a| assert!((*a).abs() <= 6.));
/// ```
pub fn random_gaussian_tensor<Float, Scalar>(
    size: usize,
    mean: Float,
    std: Float,
) -> Tensor<Vec<Scalar>>
where
    Float: FloatingPoint,
    (Scalar, Scalar): RandomGenerable<Gaussian<Float>>,
    Scalar: Numeric,
{
    let mut tensor = Tensor::allocate(Scalar::ZERO, size);
    fill_with_random_gaussian(&mut tensor, mean, std);
    tensor
}
