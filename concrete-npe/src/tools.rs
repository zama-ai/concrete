use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::numeric::UnsignedInteger;
use std::ops::Mul;

/// Computes the number of bits affected by the noise with a dispersion
/// describing a normal distribution.
pub fn estimate_number_of_noise_bits<T, D>(dispersion: D) -> usize
where
    D: DispersionParameter,
    T: UnsignedInteger,
{
    // get the standard deviation
    let std_dev: f64 = dispersion.get_modular_standard_dev::<T>();

    // the constant used for the computation
    let z: f64 = 4.;
    let tmp = f64::log2(std_dev * z);
    if tmp < 0. {
        // means no bits are affected by the noise in the integer representation
        // (discrete space)
        0usize
    } else {
        tmp.ceil() as usize
    }
}

/// Computes the square of the input value.
pub(super) fn square<T>(x: T) -> T
where
    T: Mul<T, Output = T> + Copy,
{
    x * x
}

#[cfg(test)]
pub mod tests {
    #[macro_export]
    macro_rules! assert_float_eq {
        ($given:expr, $expected:expr, $opt:ident = $eps:expr) => {
            assert!(
                ($given - $expected).abs() <= $eps,
                "{} != {} +- {}",
                $given,
                $expected,
                $eps
            );
        };
    }
    pub(crate) use assert_float_eq;
}
