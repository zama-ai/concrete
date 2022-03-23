use concrete_commons::dispersion::Variance;

pub fn from_modular_variance(modular_variance: f64, ciphertext_modulus_log: u64) -> Variance {
    match ciphertext_modulus_log {
        128 => Variance::from_modular_variance::<u128>(modular_variance),
        64 => Variance::from_modular_variance::<u64>(modular_variance),
        32 => Variance::from_modular_variance::<u32>(modular_variance),
        _ => panic!(),
    }
}
