use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    PlaintextCount, PolynomialSize,
};
use concrete_npe as npe;

use crate::backends::core::private::crypto::bootstrap::fourier::constant_sample_extract;
use crate::backends::core::private::crypto::bootstrap::{
    FourierBootstrapKey, FourierBuffers, StandardBootstrapKey,
};
use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::glwe::GlweCiphertext;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::fft::Complex64;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{AsRefTensor, IntoTensor, Tensor};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{assert_delta_std_dev, assert_noise_distribution};

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

        let mut random_generator = RandomGenerator::new(None);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

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
        let mut random_generator = RandomGenerator::new(None);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

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

fn test_sample_extract<T: UnsignedTorus>() {
    let n_tests = 10;
    // fix different polynomial degrees
    let degrees = vec![512, 1024, 2048];
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    for polynomial_size in degrees {
        // fixa set of parameters
        let mut sdk_samples = Tensor::from_container(vec![T::ZERO; n_tests]);
        let mut groundtruth_samples = Tensor::from_container(vec![T::ZERO; n_tests]);
        let std_dev = LogStandardDev(-20.);
        let dimension = GlweDimension(1);

        // compute length of the lwe secret key
        for i in 0..n_tests {
            let rlwe_sk = GlweSecretKey::generate_binary(
                dimension,
                PolynomialSize(polynomial_size),
                &mut secret_generator,
            );

            // allocate and draw a random polynomial
            let mut messages = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
            random_generator.fill_tensor_with_random_uniform(&mut messages);

            // allocate RLWE ciphertext
            let mut rlwe_ct = GlweCiphertext::allocate(
                T::ZERO,
                PolynomialSize(polynomial_size),
                dimension.to_glwe_size(),
            );
            rlwe_sk.encrypt_glwe(&mut rlwe_ct, &messages, std_dev, &mut encryption_generator);

            // allocate LWE ciphertext
            let mut lwe_ct =
                LweCiphertext::allocate(T::ZERO, LweSize(dimension.0 * polynomial_size + 1));

            // allocate space of the decrypted message (after sample extract)
            let mut new_message = Plaintext(T::ZERO);

            // perform sample extract
            constant_sample_extract(&mut lwe_ct, &rlwe_ct);

            // decrypt resulting lwe ciphertext
            let lwe_sk =
                LweSecretKey::binary_from_container(rlwe_sk.into_tensor().into_container());
            lwe_sk.decrypt_lwe(&mut new_message, &lwe_ct);
            *groundtruth_samples.get_element_mut(i) = *messages.as_tensor().get_element(0);
            *sdk_samples.get_element_mut(i) = new_message.0;
        }

        // test
        if n_tests < 7 {
            assert_delta_std_dev(&groundtruth_samples, &sdk_samples, std_dev);
        } else {
            assert_noise_distribution(&groundtruth_samples, &sdk_samples, std_dev);
        }
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

#[test]
pub fn test_sample_extract_u32() {
    test_sample_extract::<u32>();
}

#[test]
pub fn test_sample_extract_u64() {
    test_sample_extract::<u64>();
}
