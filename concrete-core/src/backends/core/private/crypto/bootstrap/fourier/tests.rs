use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PlaintextCount,
    PolynomialSize,
};
use concrete_npe as npe;

use crate::backends::core::private::crypto::bootstrap::{
    FourierBootstrapKey, FourierBuffers, StandardBootstrapKey,
};
use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::glwe::GlweCiphertext;
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::fft::Complex64;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::*;

fn test_cmux_0<T: UnsignedTorus>() {
    // fix different polynomial degrees
    let degrees = vec![512, 1024, 2048];
    for polynomial_size in degrees {
        // fix a set of parameters
        let rlwe_dimension = GlweDimension(2);
        let lwe_dimension = LweDimension(1);
        let level = DecompositionLevelCount(4);
        let base_log = DecompositionBaseLog(7);
        let std_dev_bsk = LogStandardDev(-20.);
        let std_dev_rlwe = LogStandardDev(-25.);

        let mut random_generator = new_random_generator();
        let mut secret_generator = new_secret_random_generator();
        let mut encryption_generator = new_encryption_random_generator();

        // generate the secret keys
        let rlwe_sk = GlweSecretKey::generate_binary(
            rlwe_dimension,
            PolynomialSize(polynomial_size),
            &mut secret_generator,
        );
        let lwe_sk = LweSecretKey::binary_from_container(vec![T::ZERO]);

        // create the polynomial to encrypt
        let mut m0 = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
        random_generator.fill_tensor_with_random_uniform(&mut m0);
        let mut m1 = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
        random_generator.fill_tensor_with_random_uniform(&mut m1);

        // allocate space for the decrypted polynomial
        let mut new_messages = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));

        // allocation and generation of the key in coef domain:
        let mut coef_bsk = StandardBootstrapKey::allocate(
            T::ZERO,
            rlwe_dimension.to_glwe_size(),
            PolynomialSize(polynomial_size),
            level,
            base_log,
            lwe_dimension,
        );
        coef_bsk.fill_with_new_key(&lwe_sk, &rlwe_sk, std_dev_bsk, &mut encryption_generator);

        // allocation for the bootstrapping key
        let mut fourier_bsk = FourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            rlwe_dimension.to_glwe_size(),
            PolynomialSize(polynomial_size),
            level,
            base_log,
            lwe_dimension,
        );
        let mut buffers = FourierBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
        fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);

        // allocate glwe vectors
        let mut ciphertext0 = GlweCiphertext::allocate(
            T::ZERO,
            PolynomialSize(polynomial_size),
            rlwe_dimension.to_glwe_size(),
        );
        let mut ciphertext1 = GlweCiphertext::allocate(
            T::ZERO,
            PolynomialSize(polynomial_size),
            rlwe_dimension.to_glwe_size(),
        );

        // // encrypt the polynomial
        rlwe_sk.encrypt_glwe(
            &mut ciphertext0,
            &m0,
            std_dev_rlwe,
            &mut encryption_generator,
        );
        rlwe_sk.encrypt_glwe(
            &mut ciphertext1,
            &m1,
            std_dev_rlwe,
            &mut encryption_generator,
        );

        let rgsw = fourier_bsk.ggsw_iter().next().unwrap();

        // compute cmux
        let mut buffers = FourierBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
        fourier_bsk.cmux(
            &mut ciphertext0,
            &mut ciphertext1,
            &rgsw,
            &mut buffers.fft_buffers,
            &mut buffers.rounded_buffer,
        );
        rlwe_sk.decrypt_glwe(&mut new_messages, &ciphertext0);

        // call the NPE to find the theoretical amount of noise added by the cmux
        let variance_rlwe = std_dev_rlwe.get_variance();
        let variance_trgsw = std_dev_bsk.get_variance();
        let output_variance = npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyKind>(
            rlwe_dimension,
            PolynomialSize(polynomial_size),
            base_log,
            level,
            Variance(variance_rlwe),
            Variance(variance_rlwe),
            Variance(variance_trgsw),
        );
        // test
        assert_noise_distribution(&new_messages, &m0, output_variance);
    }
}

fn test_cmux_1<T: UnsignedTorus>() {
    // fix different polynomial degrees
    let degrees = vec![512, 1024, 2048];
    for polynomial_size in degrees {
        // fix a set of parameters
        let rlwe_dimension = GlweDimension(2);
        let lwe_dimension = LweDimension(1);
        let level = DecompositionLevelCount(4);
        let base_log = DecompositionBaseLog(7);
        let std_dev_bsk = LogStandardDev(-20.);
        let std_dev_rlwe = LogStandardDev(-25.);
        let mut random_generator = new_random_generator();
        let mut secret_generator = new_secret_random_generator();
        let mut encryption_generator = new_encryption_random_generator();

        // generate the secret keys
        let rlwe_sk = GlweSecretKey::generate_binary(
            rlwe_dimension,
            PolynomialSize(polynomial_size),
            &mut secret_generator,
        );
        let lwe_sk = LweSecretKey::binary_from_container(vec![T::ONE]);

        // create the polynomial to encrypt
        let mut m0 = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
        random_generator.fill_tensor_with_random_uniform(&mut m0);
        let mut m1 = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
        random_generator.fill_tensor_with_random_uniform(&mut m1);

        // allocate space for the decrypted polynomial
        let mut new_messages = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));

        // allocation and generation of the key in coef domain:
        let mut coef_bsk = StandardBootstrapKey::allocate(
            T::ZERO,
            rlwe_dimension.to_glwe_size(),
            PolynomialSize(polynomial_size),
            level,
            base_log,
            lwe_dimension,
        );
        coef_bsk.fill_with_new_key(&lwe_sk, &rlwe_sk, std_dev_bsk, &mut encryption_generator);

        // allocation for the bootstrapping key
        let mut fourier_bsk = FourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            rlwe_dimension.to_glwe_size(),
            PolynomialSize(polynomial_size),
            level,
            base_log,
            lwe_dimension,
        );
        let mut buffers = FourierBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
        fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);

        // allocate glwe vectors
        let mut ciphertext0 = GlweCiphertext::allocate(
            T::ZERO,
            PolynomialSize(polynomial_size),
            rlwe_dimension.to_glwe_size(),
        );
        let mut ciphertext1 = GlweCiphertext::allocate(
            T::ZERO,
            PolynomialSize(polynomial_size),
            rlwe_dimension.to_glwe_size(),
        );

        // // encrypt the polynomial
        rlwe_sk.encrypt_glwe(
            &mut ciphertext0,
            &m0,
            std_dev_rlwe,
            &mut encryption_generator,
        );
        rlwe_sk.encrypt_glwe(
            &mut ciphertext1,
            &m1,
            std_dev_rlwe,
            &mut encryption_generator,
        );

        let rgsw = fourier_bsk.ggsw_iter().next().unwrap();

        // compute cmux
        let mut buffers = FourierBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
        fourier_bsk.cmux(
            &mut ciphertext0,
            &mut ciphertext1,
            &rgsw,
            &mut buffers.fft_buffers,
            &mut buffers.rounded_buffer,
        );
        rlwe_sk.decrypt_glwe(&mut new_messages, &ciphertext0);

        // call the NPE to find the theoretical amount of noise added by the cmux
        let variance_rlwe = std_dev_rlwe.get_variance();
        let variance_trgsw = std_dev_bsk.get_variance();
        let output_variance = npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyKind>(
            rlwe_dimension,
            PolynomialSize(polynomial_size),
            base_log,
            level,
            Variance(variance_rlwe),
            Variance(variance_rlwe),
            Variance(variance_trgsw),
        );
        // test
        assert_noise_distribution(&new_messages, &m1, output_variance);
    }
}

#[test]
pub fn test_cmux0_u32() {
    test_cmux_0::<u32>();
}

#[test]
pub fn test_cmux0_u64() {
    test_cmux_0::<u64>();
}

#[test]
pub fn test_cmux_1_u32() {
    test_cmux_1::<u32>();
}

#[test]
pub fn test_cmux_1_u64() {
    test_cmux_1::<u64>();
}
