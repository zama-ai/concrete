use concrete_commons::dispersion::DispersionParameter;

/// Computes the number of bits affected by the noise with a dispersion
/// describing a normal distribution
/// Arguments:
/// * `dispersion` - noise variance of the ciphertext
/// * `log_integer_modulus`- the log_2 of the integer modulus q
/// Output:
/// * The number of bit affected by the noise
pub fn nb_bit_from_variance<D>(dispersion: D, log_integer_modulus: usize) -> usize
where
    D: DispersionParameter,
{
    // get the standard deviation
    let std_dev: f64 = dispersion.get_standard_dev();

    // the constant used for the computation
    let z: f64 = 4.;
    let tmp = log_integer_modulus as f64 + f64::log2(std_dev * z);
    if tmp < 0. {
        // means no bits are affected by the noise in the integer representation
        // (discrete space)
        0usize
    } else {
        tmp.ceil() as usize
    }
}
