use concrete_commons::dispersion::LogStandardDev;

use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::Tensor;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::assert_noise_distribution;

fn test_normal_random<T: UnsignedTorus>() {
    //! test if the normal random generation with std_dev is below 3*std_dev (99.7%)

    // settings
    let std_dev: f64 = f64::powi(2., -20);
    let mean: f64 = 0.;
    let k = 1_000_000;
    let mut generator = RandomGenerator::new(None);

    // generates normal random
    let mut samples_int = Tensor::allocate(T::ZERO, k);
    generator.fill_tensor_with_random_gaussian(&mut samples_int, mean, std_dev);

    // converts into float
    let mut samples_float = Tensor::allocate(0f64, k);
    samples_float.fill_with_one(&samples_int, |a| a.into_torus());
    for x in samples_float.iter_mut() {
        if *x > 0.5 {
            *x = 1. - *x;
        }
    }

    // tests if over 3*std_dev
    let mut number_of_samples_outside_confidence_interval: usize = 0;
    for s in samples_float.iter() {
        if *s > 3. * std_dev || *s < -3. * std_dev {
            number_of_samples_outside_confidence_interval += 1;
        }
    }

    // computes the percentage of samples over 3*std_dev
    let proportion_of_samples_outside_confidence_interval: f64 =
        (number_of_samples_outside_confidence_interval as f64) / (k as f64);

    // test
    assert!(
        proportion_of_samples_outside_confidence_interval < 0.003,
        "test normal random : proportion = {} ; n = {}",
        proportion_of_samples_outside_confidence_interval,
        number_of_samples_outside_confidence_interval
    );
}

#[test]
fn test_normal_random_u32() {
    test_normal_random::<u32>();
}

#[test]
fn test_normal_random_u64() {
    test_normal_random::<u64>();
}

fn test_distribution<T: UnsignedTorus>() {
    //! tests gaussianity against the rand crate generation
    // settings
    let std_dev: f64 = f64::powi(2., -5);
    let mean: f64 = 0.;
    let k = 1_000_000;
    let mut generator = RandomGenerator::new(None);

    // generates normal random
    let first = Tensor::allocate(T::ZERO, k);
    let mut second = Tensor::allocate(T::ZERO, k);
    generator.fill_tensor_with_random_gaussian(&mut second, mean, std_dev);

    assert_noise_distribution(&first, &second, LogStandardDev(-5.));
}

#[test]
fn test_distribution_u32() {
    test_distribution::<u32>();
}

#[test]
fn test_distribution_u64() {
    test_distribution::<u64>();
}
