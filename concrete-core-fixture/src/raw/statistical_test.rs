//! A module containing statistical testing entry points for raw integers
use crate::raw::generation::RawUnsignedIntegers;
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_core::backends::core::private::math::random::RandomGenerator;
use kolmogorov_smirnov;

/// A function performing a Kolmogorov Smirnov statistical test.
///
/// The `tested` argument points to an array of samples which is tested for normality. The
/// `expected_means` argument encodes the mean expected for each element of the `tested` array (can
/// be different for each element). The `expected_variance` argument encodes the variance expected
/// for the whole sample array.
#[allow(dead_code)]
pub fn assert_noise_distribution<Raw>(
    tested: &[Raw],
    expected_means: &[Raw],
    expected_variance: Variance,
) -> bool
where
    Raw: RawUnsignedIntegers,
{
    let std_dev = expected_variance.get_standard_dev();
    let confidence = 0.95;
    let n_slots = expected_means.len();
    let mut generator = RandomGenerator::new(None);

    // allocate 2 slices: one for the error samples obtained, the second for fresh samples
    // according to the std_dev computed
    let mut sdk_samples = vec![0.0_f64; n_slots];

    // recover the errors from each ciphertexts
    sdk_samples
        .iter_mut()
        .zip(expected_means.iter())
        .zip(tested.iter())
        .for_each(|((sample, first), second)| *sample = torus_modular_distance(*first, *second));

    // fill the theoretical sample vector according to std_dev
    let theoretical_samples = generator
        .random_gaussian_tensor(n_slots, 0., std_dev)
        .into_container();

    // compute the Kolmogorov Smirnov test
    let result = kolmogorov_smirnov::test_f64(
        sdk_samples.as_slice(),
        theoretical_samples.as_slice(),
        confidence,
    );

    !result.is_rejected | {
        // compute the mean of our errors
        let mut mean: f64 = sdk_samples.iter().sum();
        mean /= sdk_samples.len() as f64;

        // compute the variance of the errors
        let mut sdk_variance: f64 = sdk_samples.iter().map(|x| f64::powi(x - mean, 2)).sum();
        sdk_variance /= (sdk_samples.len() - 1) as f64;

        // compute the standard deviation
        let sdk_std_log2 = f64::log2(f64::sqrt(sdk_variance));
        let th_std_log2 = f64::log2(std_dev);

        // test if theoretical_std_dev > sdk_std_dev
        if sdk_std_log2 > th_std_log2 {
            println!("sdk std {:?}, th std {:?}", sdk_std_log2, th_std_log2);
        }
        // Here due to rounding issues we give a 0.5 "slack" to the comparison
        // The 0.5 value itself is arbitrary (we chose it to correspond to a rounding)
        // and may be changed. Actually we plan to re-work the statistical test
        // which we hope will remove the need for this.
        sdk_std_log2 <= th_std_log2 + 0.5
    }
}

pub fn assert_delta_std_dev<Raw>(
    first: &[Raw],
    second: &[Raw],
    dist: impl DispersionParameter,
) -> bool
where
    Raw: RawUnsignedIntegers,
{
    for (x, y) in first.iter().zip(second.iter()) {
        let distance: f64 = torus_modular_distance(*x, *y);
        let torus_distance = distance / 2_f64.powi(Raw::BITS as i32);
        if torus_distance > 5. * dist.get_standard_dev() {
            return false;
        }
    }
    true
}

fn torus_modular_distance<T: RawUnsignedIntegers>(first: T, other: T) -> f64 {
    let d0 = first.wrapping_sub(other);
    let d1 = other.wrapping_sub(first);
    if d0 < d1 {
        let d: f64 = d0.cast_into();
        d / 2_f64.powi(T::BITS as i32)
    } else {
        let d: f64 = d1.cast_into();
        -d / 2_f64.powi(T::BITS as i32)
    }
}
