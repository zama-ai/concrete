use crate::backends::core::private::crypto::encoding::Plaintext;
use crate::backends::core::private::crypto::gsw::GswCiphertext;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{AsMutSlice, Tensor};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::assert_noise_distribution;
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::markers::BinaryKeyDistribution;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_npe as npe;

fn test_external_product_gsw<T: UnsignedTorus>() {
    let n_tests = 10;

    // allocate message vectors
    let mut msg = Tensor::allocate(T::ZERO, n_tests);
    let mut new_msg = Tensor::allocate(T::ZERO, n_tests);

    // fix a set of parameters
    let dimension = LweDimension(630);
    let level = DecompositionLevelCount(6);
    let base_log = DecompositionBaseLog(4);
    let std_dev = LogStandardDev(-20.);
    for i in 0..n_tests {
        // We instantiate the random generators.
        let mut random_generator = RandomGenerator::new(None);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

        // generate the lwe secret key
        let lwe_sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

        // create the message to encrypt
        let message = Plaintext(random_generator.random_uniform());

        // create the GSW multiplication factor
        let mul_u8: u8 = random_generator.random_uniform_n_lsb(1);
        let mul = match mul_u8 {
            1 => T::ONE,
            _ => T::ZERO,
        };
        msg.as_mut_slice()[i] = message.0 * mul;

        // allocation and generation of the GSW
        let mut gsw = GswCiphertext::allocate(T::ZERO, dimension.to_lwe_size(), level, base_log);
        lwe_sk.encrypt_constant_gsw(
            &mut gsw,
            &Plaintext(mul),
            std_dev,
            &mut encryption_generator,
        );

        // allocate space for the decrypted message
        let mut new_message = Plaintext(T::ZERO);

        // allocate vectors for lwe ciphertexts (inputs)
        let mut ciphertext = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());

        // allocate vectors for lwe ciphertexts (outputs)
        let mut res = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());

        // encrypt the message
        lwe_sk.encrypt_lwe(
            &mut ciphertext,
            &message,
            std_dev,
            &mut encryption_generator,
        );

        gsw.external_product(&mut res, &ciphertext);

        lwe_sk.decrypt_lwe(&mut new_message, &res);
        new_msg.as_mut_slice()[i] = new_message.0;
    }

    // call the NPE to find the theoretical amount of noise after the bootstrap
    let output_variance =
        npe::estimate_external_product_noise_with_binary_ggsw::<T, _, _, BinaryKeyDistribution>(
            PolynomialSize(1),
            GlweDimension(dimension.0),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            base_log,
            level,
        );
    // we check that the obtain distribution is the same
    // as the theoretical one
    assert_noise_distribution(&msg, &new_msg, output_variance);
}

fn test_cmux_0_gsw<T: UnsignedTorus>() {
    let n_tests = 10;

    // allocate message vectors
    let mut msg = Tensor::allocate(T::ZERO, n_tests);
    let mut new_msg = Tensor::allocate(T::ZERO, n_tests);

    // fix a set of parameters
    let dimension = LweDimension(630);
    let level = DecompositionLevelCount(4);
    let base_log = DecompositionBaseLog(7);
    let std_dev = LogStandardDev(-25.);
    for i in 0..n_tests {
        let mut random_generator = RandomGenerator::new(None);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

        // generate the secret key
        let lwe_sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

        // create the messages to encrypt
        let m0 = Plaintext(random_generator.random_uniform());
        let m1 = Plaintext(random_generator.random_uniform());
        msg.as_mut_slice()[i] = m0.0;

        // allocate space for the decrypted message
        let mut new_message = Plaintext(T::ZERO);

        // allocation and generation of the GSW
        let mut gsw = GswCiphertext::allocate(T::ZERO, dimension.to_lwe_size(), level, base_log);
        lwe_sk.encrypt_constant_gsw(
            &mut gsw,
            &Plaintext(T::ZERO),
            std_dev,
            &mut encryption_generator,
        );

        // allocate lwe vectors
        let mut ciphertext0 = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        let mut ciphertext1 = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        let mut out = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());

        // encrypt the messages
        lwe_sk.encrypt_lwe(&mut ciphertext0, &m0, std_dev, &mut encryption_generator);
        lwe_sk.encrypt_lwe(&mut ciphertext1, &m1, std_dev, &mut encryption_generator);

        // compute cmux
        gsw.cmux(&mut out, &ciphertext0, &ciphertext1);
        lwe_sk.decrypt_lwe(&mut new_message, &out);
        new_msg.as_mut_slice()[i] = new_message.0;
    }

    // call the NPE to find the theoretical amount of noise after the cmux
    let output_variance =
        npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyDistribution>(
            GlweDimension(dimension.0),
            PolynomialSize(1),
            base_log,
            level,
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
        );
    // we check that the obtain distribution is the same
    // as the theoretical one
    assert_noise_distribution(&msg, &new_msg, output_variance);
}

fn test_cmux_1_gsw<T: UnsignedTorus>() {
    let n_tests = 10;

    // allocate message vectors
    let mut msg = Tensor::allocate(T::ZERO, n_tests);
    let mut new_msg = Tensor::allocate(T::ZERO, n_tests);

    // fix a set of parameters
    let dimension = LweDimension(630);
    let level = DecompositionLevelCount(4);
    let base_log = DecompositionBaseLog(7);
    let std_dev = LogStandardDev(-25.);
    for i in 0..n_tests {
        let mut random_generator = RandomGenerator::new(None);
        let mut secret_generator = SecretRandomGenerator::new(None);
        let mut encryption_generator = EncryptionRandomGenerator::new(None);

        // generate the secret key
        let lwe_sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

        // create the messages to encrypt
        let m0 = Plaintext(random_generator.random_uniform());
        let m1 = Plaintext(random_generator.random_uniform());
        msg.as_mut_slice()[i] = m1.0;

        // allocate space for the decrypted message
        let mut new_message = Plaintext(T::ZERO);

        // allocation and generation of the GSW
        let mut gsw = GswCiphertext::allocate(T::ZERO, dimension.to_lwe_size(), level, base_log);
        lwe_sk.encrypt_constant_gsw(
            &mut gsw,
            &Plaintext(T::ONE),
            std_dev,
            &mut encryption_generator,
        );

        // allocate lwe vectors
        let mut ciphertext0 = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        let mut ciphertext1 = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        let mut out = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());

        // encrypt the messages
        lwe_sk.encrypt_lwe(&mut ciphertext0, &m0, std_dev, &mut encryption_generator);
        lwe_sk.encrypt_lwe(&mut ciphertext1, &m1, std_dev, &mut encryption_generator);

        // compute cmux
        gsw.cmux(&mut out, &ciphertext0, &ciphertext1);
        lwe_sk.decrypt_lwe(&mut new_message, &out);
        new_msg.as_mut_slice()[i] = new_message.0;
    }

    // call the NPE to find the theoretical amount of noise after the bootstrap
    let output_variance =
        npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyDistribution>(
            GlweDimension(dimension.0),
            PolynomialSize(1),
            base_log,
            level,
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
            Variance(f64::powi(std_dev.get_standard_dev(), 2)),
        );
    // we check that the obtain distribution is the same
    // as the theoretical one
    assert_noise_distribution(&msg, &new_msg, output_variance);
}

#[test]
pub fn test_external_product_gsw_u32() {
    test_external_product_gsw::<u32>()
}

#[test]
pub fn test_external_product_gsw_u64() {
    test_external_product_gsw::<u64>()
}

#[test]
pub fn test_cmux0_gsw_u32() {
    test_cmux_0_gsw::<u32>();
}

#[test]
pub fn test_cmux0_gsw_u64() {
    test_cmux_0_gsw::<u64>();
}

#[test]
pub fn test_cmux_1_gsw_u32() {
    test_cmux_1_gsw::<u32>();
}

#[test]
pub fn test_cmux_1_gsw_u64() {
    test_cmux_1_gsw::<u64>();
}
