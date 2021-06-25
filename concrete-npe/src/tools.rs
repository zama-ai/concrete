use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::numeric::UnsignedInteger;
use std::ops::Mul;

/// Computes the number of bits affected by the noise with a dispersion
/// describing a normal distribution
/// Arguments:
/// * `dispersion` - noise variance of the ciphertext
/// * `log_integer_modulus`- the log_2 of the integer modulus q
/// Output:
/// * The number of bit affected by the noise
pub fn nb_bit_from_variance<T, D>(dispersion: D) -> usize
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

/// Square function tool
/// Arguments:
/// * `x` - input
/// Output
/// * x^2
pub fn square<T>(x: T) -> T
where
    T: Mul<T, Output = T> + Copy,
{
    x * x
}
