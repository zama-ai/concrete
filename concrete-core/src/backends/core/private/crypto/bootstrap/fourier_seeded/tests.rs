use concrete_commons::{
    dispersion::LogStandardDev,
    parameters::{DecompositionBaseLog, DecompositionLevelCount, PolynomialSize},
};

use crate::backends::core::private::{
    crypto::bootstrap::FourierBuffers, math::torus::UnsignedTorus,
};
use crate::backends::core::private::{crypto::bootstrap::StandardSeededBootstrapKey, test_tools};
use crate::backends::core::private::{
    crypto::{bootstrap::FourierBootstrapKey, secret::generators::SecretRandomGenerator},
    math::fft::Complex64,
};
use crate::backends::core::private::{
    crypto::{
        bootstrap::StandardBootstrapKey,
        secret::{GlweSecretKey, LweSecretKey},
    },
    math::tensor::{AsRefSlice, AsRefTensor},
};

use super::FourierSeededBootstrapKey;

fn test_fourier_seeded_bootstrap<T: UnsignedTorus>() {
    // random settings
    let glwe_dimension = test_tools::random_glwe_dimension(5);
    let lwe_dimension = test_tools::random_lwe_dimension(100);
    let polynomial_size = test_tools::random_polynomial_size(4);
    let polynomial_size = PolynomialSize(1 << (7 + polynomial_size.0));
    let noise_parameters = LogStandardDev::from_log_standard_dev(-50.);
    let decomp_level = DecompositionLevelCount(3);
    let decomp_base_log = DecompositionBaseLog(7);
    let mut secret_generator = SecretRandomGenerator::new(None);

    // generates a secret key
    let glwe_sk =
        GlweSecretKey::generate_binary(glwe_dimension, polynomial_size, &mut secret_generator);

    let lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);

    // allocation and generation of the key in coef domain:
    let mut coef_bsk_seeded = StandardSeededBootstrapKey::allocate(
        T::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    coef_bsk_seeded.fill_with_new_key(&lwe_sk, &glwe_sk, noise_parameters);

    // allocation for the bootstrapping key
    let mut buffers = FourierBuffers::new(polynomial_size, glwe_dimension.to_glwe_size());
    let mut fourier_bsk_seeded = FourierSeededBootstrapKey::allocate(
        Complex64::new(0., 0.),
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    fourier_bsk_seeded.fill_with_forward_fourier(&coef_bsk_seeded, &mut buffers);

    // expansion of the bootstrapping key
    let mut fourier_bsk_expanded = FourierBootstrapKey::allocate(
        Complex64::new(0., 0.),
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    fourier_bsk_seeded.expand_into(&mut fourier_bsk_expanded, &mut buffers);

    //control
    let mut coef_bsk_expanded = StandardBootstrapKey::allocate(
        T::ZERO,
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    coef_bsk_seeded.expand_into(&mut coef_bsk_expanded);
    let mut fourier_bsk_control = FourierBootstrapKey::allocate(
        Complex64::new(0., 0.),
        glwe_dimension.to_glwe_size(),
        polynomial_size,
        decomp_level,
        decomp_base_log,
        lwe_dimension,
    );
    fourier_bsk_control.fill_with_forward_fourier(&coef_bsk_expanded, &mut buffers);

    assert_eq!(
        fourier_bsk_control.as_tensor().as_slice(),
        fourier_bsk_expanded.as_tensor().as_slice()
    );
}

#[test]
fn test_fourier_seeded_bootstrap_u32() {
    test_fourier_seeded_bootstrap::<u32>()
}

#[test]
fn test_fourier_seeded_bootstrap_u64() {
    test_fourier_seeded_bootstrap::<u64>()
}
