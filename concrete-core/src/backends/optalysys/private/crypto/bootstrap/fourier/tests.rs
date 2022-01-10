use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::{CastFrom, CastInto, Numeric};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    PlaintextCount, PolynomialSize,
};
use concrete_npe as npe;
use std::fmt::Debug;

use crate::backends::core::private::crypto::bootstrap::fourier::constant_sample_extract;
use crate::backends::core::private::crypto::bootstrap::StandardBootstrapKey;
use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::glwe::GlweCiphertext;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::fft::Complex64;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{
    AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{assert_delta_std_dev, assert_noise_distribution};
use crate::backends::optalysys::private::crypto::bootstrap::fourier::{
    FourierBootstrapKey, FourierBskBuffers,
};

fn test_bootstrap_noise<T: UnsignedTorus>() {
    //! test that the bootstrapping noise matches the theoretical noise
    //! This test is design to remove the impact of the drift, we only
    //! check the noise added by the external products

    for size in &[512, 1024, 2048] {
        // fix a set of parameters
        let nb_test: usize = 2;
        let polynomial_size = PolynomialSize(*size);
        let rlwe_dimension = GlweDimension(1);
        let lwe_dimension = LweDimension(630);
        let level = DecompositionLevelCount(3);
        let base_log = DecompositionBaseLog(7);
        let std = LogStandardDev::from_log_standard_dev(-29.);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

        // allocate message vectors
        let mut msg = Tensor::allocate(T::ZERO, nb_test);
        let mut new_msg = Tensor::allocate(T::ZERO, nb_test);

        // launch nb_test tests
        for i in 0..nb_test {
            // allocate secret keys
            let rlwe_sk = GlweSecretKey::generate_binary(
                rlwe_dimension,
                polynomial_size,
                &mut secret_generator,
            );
            let lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);

            // allocation and generation of the key in coef domain:
            let mut coef_bsk = StandardBootstrapKey::allocate(
                T::ZERO,
                rlwe_dimension.to_glwe_size(),
                polynomial_size,
                level,
                base_log,
                lwe_dimension,
            );
            coef_bsk.fill_with_new_key(&lwe_sk, &rlwe_sk, std, &mut encryption_generator);

            // allocation for the bootstrapping key
            let mut fourier_bsk = FourierBootstrapKey::allocate(
                Complex64::new(0., 0.),
                rlwe_dimension.to_glwe_size(),
                polynomial_size,
                level,
                base_log,
                lwe_dimension,
            );
            let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
            fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);

            // Create a fix message (encoded in the most significant bit of the torus)
            // put a 3 bit message XXX here 0XXX000...000 in the torus bit representation
            let mut random_generator = RandomGenerator::new(None);
            let random: T = random_generator.random_uniform();
            let m0 = Plaintext(random & (((T::ONE << 3) - T::ONE) << (T::BITS - 4)));
            println!("{:?}", m0.0);

            // allocate ciphertext vectors
            let mut lwe_in = LweCiphertext::allocate(T::ZERO, lwe_dimension.to_lwe_size());
            let mut lwe_out =
                LweCiphertext::allocate(T::ZERO, LweSize(rlwe_dimension.0 * polynomial_size.0 + 1));
            lwe_sk.encrypt_lwe(&mut lwe_in, &m0, std, &mut encryption_generator);
            // Here we define a LUT that takes a message as an input
            // and outputs:
            // -1/8 if the most significant bit of the message is one
            // 1/8 otherwise
            let cst = T::ONE << (T::BITS - 3);
            msg.as_mut_slice()[i] = cst;

            // create a constant accumulator
            let mut accumulator =
                GlweCiphertext::allocate(T::ZERO, polynomial_size, rlwe_dimension.to_glwe_size());
            accumulator
                .get_mut_body()
                .as_mut_tensor()
                .fill_with_element(cst);

            let mut buffers =
                FourierBskBuffers::new(fourier_bsk.polynomial_size(), fourier_bsk.glwe_size());
            fourier_bsk.bootstrap(&mut lwe_out, &lwe_in, &accumulator, &mut buffers);

            // Initialize a m1 message
            let mut m1 = Plaintext(T::ZERO);

            // now the lwe is decrypted using a flatten of the rlwe encryption key
            let flattened_key = LweSecretKey::binary_from_container(rlwe_sk.as_tensor().as_slice());
            flattened_key.decrypt_lwe(&mut m1, &lwe_out);
            // store the decryption of the bootstrapped ciphertext
            new_msg.as_mut_slice()[i] = m1.0;
        }

        // call the NPE to find the theoretical amount of noise after the bootstrap
        let output_variance = npe::estimate_pbs_noise::<T, Variance, BinaryKeyKind>(
            lwe_dimension,
            polynomial_size,
            rlwe_dimension,
            base_log,
            level,
            Variance(f64::powi(std.get_standard_dev(), 2)),
        );
        // if we have enough test, we check that the obtain distribution is the same
        // as the theoretical one
        // if not, it only tests if the noise remains in the 99% confidence interval
        if nb_test < 7 {
            assert_delta_std_dev(&msg, &new_msg, output_variance);
        } else {
            assert_noise_distribution(&msg, &new_msg, output_variance);
        }
    }
}

fn test_external_product_generic<T: UnsignedTorus>() {
    let n_tests = 10;
    for _n in 0..n_tests {
        // fix different polynomial degrees
        let degrees = vec![512, 1024, 2048];
        for polynomial_size in degrees {
            // fix a set of parameters
            let rlwe_dimension = GlweDimension(2);
            let lwe_dimension = LweDimension(1);
            let level = DecompositionLevelCount(6);
            let base_log = DecompositionBaseLog(4);
            let std_dev_bsk = LogStandardDev(-25.);
            let std_dev_rlwe = LogStandardDev(-20.);

            // We instantiate the random generators.
            let mut random_generator = RandomGenerator::new(None);
            let mut secret_generator = SecretRandomGenerator::new(None);
            let mut encryption_generator = EncryptionRandomGenerator::new(None);

            // compute the length of glwe secret key
            let rlwe_sk = GlweSecretKey::generate_binary(
                rlwe_dimension,
                PolynomialSize(polynomial_size),
                &mut secret_generator,
            );

            // We create a lwe secret key with one bit set to one
            let lwe_sk = LweSecretKey::binary_from_container(vec![T::ONE]);

            // create the polynomial to encrypt
            let mut messages = PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));
            random_generator.fill_tensor_with_random_uniform(&mut messages);

            // allocate space for the decrypted polynomial
            let mut new_messages =
                PlaintextList::allocate(T::ZERO, PlaintextCount(polynomial_size));

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
            let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
            fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);

            // allocate vectors for glwe ciphertexts (inputs)
            let mut ciphertext = GlweCiphertext::allocate(
                T::ZERO,
                PolynomialSize(polynomial_size),
                rlwe_dimension.to_glwe_size(),
            );

            // allocate vectors for glwe ciphertexts (outputs)
            let mut res = GlweCiphertext::allocate(
                T::ZERO,
                PolynomialSize(polynomial_size),
                rlwe_dimension.to_glwe_size(),
            );

            // // encrypt the polynomial
            rlwe_sk.encrypt_glwe(
                &mut ciphertext,
                &messages,
                std_dev_rlwe,
                &mut encryption_generator,
            );
            let rgsw = fourier_bsk.ggsw_iter().next().unwrap();

            let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
            fourier_bsk.external_product(
                &mut res,
                &rgsw,
                &ciphertext,
                &mut buffers.fft_buffers,
                &mut buffers.rounded_buffer,
            );

            rlwe_sk.decrypt_glwe(&mut new_messages, &res);

            // call the NPE to find the theoretical amount of noise after the external product
            let var_trgsw = std_dev_bsk.get_variance();
            let var_trlwe = std_dev_rlwe.get_variance();
            let output_variance = npe::estimate_external_product_noise_with_binary_ggsw::<
                T,
                Variance,
                Variance,
                BinaryKeyKind,
            >(
                PolynomialSize(polynomial_size),
                rlwe_dimension,
                Variance(var_trlwe),
                Variance(var_trgsw),
                base_log,
                level,
            );

            // test
            assert_noise_distribution(&new_messages, &messages, output_variance);
        }
    }
}

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
        let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
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
        let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
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
        let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
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
        let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
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

fn test_bootstrap_drift<T: UnsignedTorus + Debug>()
where
    i64: CastFrom<T>,
{
    // define settings
    let nb_test: usize = 10;
    let polynomial_size = PolynomialSize(1024);
    let rlwe_dimension = GlweDimension(1);
    let lwe_dimension = LweDimension(630);
    let level = DecompositionLevelCount(3);
    let base_log = DecompositionBaseLog(7);
    let std = LogStandardDev::from_log_standard_dev(-29.);
    let log_degree = f64::log2(polynomial_size.0 as f64) as i32;
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    let mut rlwe_sk =
        GlweSecretKey::generate_binary(rlwe_dimension, polynomial_size, &mut secret_generator);
    let mut lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);

    let mut msg = Tensor::allocate(T::ZERO, nb_test);
    let mut new_msg = Tensor::allocate(T::ZERO, nb_test);

    // launch nb_test tests
    for i in 0..nb_test {
        // fill keys with random
        random_generator.fill_tensor_with_random_uniform_binary(&mut rlwe_sk);
        random_generator.fill_tensor_with_random_uniform_binary(&mut lwe_sk);

        // allocation and generation of the key in coef domain:
        let mut coef_bsk = StandardBootstrapKey::allocate(
            T::ZERO,
            rlwe_dimension.to_glwe_size(),
            polynomial_size,
            level,
            base_log,
            lwe_dimension,
        );
        coef_bsk.fill_with_new_key(&lwe_sk, &rlwe_sk, std, &mut encryption_generator);

        // allocation for the bootstrapping key
        let mut fourier_bsk = FourierBootstrapKey::allocate(
            Complex64::new(0., 0.),
            rlwe_dimension.to_glwe_size(),
            polynomial_size,
            level,
            base_log,
            lwe_dimension,
        );
        let mut buffers = FourierBskBuffers::new(fourier_bsk.poly_size, fourier_bsk.glwe_size);
        fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);

        let val = (polynomial_size.0 as f64 - (10. * f64::sqrt((lwe_dimension.0 as f64) / 16.0)))
            * 2_f64.powi(<T as Numeric>::BITS as i32 - log_degree - 1);
        let val = T::cast_from(val);

        let m0 = Plaintext(val);

        msg.as_mut_slice()[i] = val;

        let mut lwe_in = LweCiphertext::allocate(T::ZERO, lwe_dimension.to_lwe_size());
        let mut lwe_out =
            LweCiphertext::allocate(T::ZERO, LweSize(rlwe_dimension.0 * polynomial_size.0 + 1));
        lwe_sk.encrypt_lwe(&mut lwe_in, &m0, std, &mut encryption_generator);

        // accumulator is a trivial encryption of [0, 1/2N, 2/2N, ...]
        let mut accumulator =
            GlweCiphertext::allocate(T::ZERO, polynomial_size, rlwe_dimension.to_glwe_size());
        accumulator
            .get_mut_body()
            .as_mut_tensor()
            .iter_mut()
            .enumerate()
            .for_each(|(i, a)| {
                *a = (i as f64 * 2_f64.powi(<T as Numeric>::BITS as i32 - log_degree - 1))
                    .cast_into();
            });

        // bootstrap
        let mut buffers =
            FourierBskBuffers::new(fourier_bsk.polynomial_size(), fourier_bsk.glwe_size());
        fourier_bsk.bootstrap(&mut lwe_out, &lwe_in, &accumulator, &mut buffers);

        let mut m1 = Plaintext(T::ZERO);

        // now the lwe is encrypted using a flatten of the trlwe encryption key
        let flattened_key = LweSecretKey::binary_from_container(rlwe_sk.as_tensor().as_slice());
        flattened_key.decrypt_lwe(&mut m1, &lwe_out);
        // store the decryption of the bootstrapped ciphertext
        new_msg.as_mut_slice()[i] = m1.0;

        // test that the drift remains within the bound of the theretical drift
        let delta_max: i64 = ((5. * f64::sqrt((lwe_dimension.0 as f64) / 16.0))
            * 2_f64.powi(<T as Numeric>::BITS as i32 - log_degree - 1))
            as i64;
        assert!(
            (i64::cast_from(m0.0) - i64::cast_from(m1.0)).abs() <= delta_max,
            "{:?} != {:?} +- {:?}",
            m0.0,
            m1.0,
            delta_max
        );
    }
}

#[test]
pub fn test_bootstrap_drift_u32() {
    test_bootstrap_drift::<u32>();
}

#[test]
pub fn test_bootstrap_drift_u64() {
    test_bootstrap_drift::<u64>();
}

#[test]
pub fn test_bootstrap_noise_u32() {
    test_bootstrap_noise::<u32>()
}

#[test]
pub fn test_bootstrap_noise_u64() {
    test_bootstrap_noise::<u64>()
}

#[test]
pub fn test_external_product_generic_u32() {
    test_external_product_generic::<u32>()
}

#[test]
pub fn test_external_product_generic_u64() {
    test_external_product_generic::<u64>()
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
