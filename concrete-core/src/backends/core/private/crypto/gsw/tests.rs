use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::gsw::GswCiphertext;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{AsMutSlice, Tensor};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{self, assert_noise_distribution};
use concrete_commons::dispersion::{DispersionParameter, LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PlaintextCount,
    PolynomialSize,
};
use concrete_npe as npe;

use super::GswSeededCiphertext;

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
        npe::estimate_external_product_noise_with_binary_ggsw::<T, _, _, BinaryKeyKind>(
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
    let output_variance = npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyKind>(
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
    let output_variance = npe::estimate_cmux_noise_with_binary_ggsw::<T, _, _, _, BinaryKeyKind>(
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

fn test_seeded_gsw<T: UnsignedTorus>() {
    // random settings
    let nb_ct = test_tools::random_ciphertext_count(10);
    let dimension = test_tools::random_lwe_dimension(5);
    let noise_parameters = LogStandardDev::from_log_standard_dev(-50.);
    let decomp_level = DecompositionLevelCount(3);
    let decomp_base_log = DecompositionBaseLog(7);
    let mut seed_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // generates a secret key
    let sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

    // generates random plaintexts
    let plaintext_vector =
        PlaintextList::from_tensor(secret_generator.random_uniform_tensor(nb_ct.0));
    let mut seeded_decryptions =
        PlaintextList::from_container(vec![T::ZERO; nb_ct.0 * decomp_level.0 * (dimension.0 + 1)]);
    let mut control_decryptions =
        PlaintextList::from_container(vec![T::ZERO; nb_ct.0 * decomp_level.0 * (dimension.0 + 1)]);

    for (plaintext, (mut seeded_decryption, mut control_decryption)) in
        plaintext_vector.plaintext_iter().zip(
            seeded_decryptions
                .sublist_iter_mut(PlaintextCount(decomp_level.0 * (dimension.0 + 1)))
                .zip(
                    control_decryptions
                        .sublist_iter_mut(PlaintextCount(decomp_level.0 * (dimension.0 + 1))),
                ),
        )
    {
        // encrypts
        let mut seeded_gsw = GswSeededCiphertext::allocate(
            T::ZERO,
            dimension.to_lwe_size(),
            decomp_level,
            decomp_base_log,
        );
        let seed = seed_generator.generate_seed();
        sk.encrypt_constant_seeded_gsw(&mut seeded_gsw, plaintext, noise_parameters, seed);

        // expands
        let mut gsw_expanded = GswCiphertext::allocate(
            T::ZERO,
            dimension.to_lwe_size(),
            decomp_level,
            decomp_base_log,
        );
        seeded_gsw.expand_into(&mut gsw_expanded);

        // control encryption
        let mut gsw = GswCiphertext::allocate(
            T::ZERO,
            dimension.to_lwe_size(),
            decomp_level,
            decomp_base_log,
        );
        sk.encrypt_constant_gsw(
            &mut gsw,
            plaintext,
            noise_parameters,
            &mut encryption_generator,
        );

        for ((seeded_lwe, control_lwe), (decoded_seeded, decoded_control)) in gsw_expanded
            .as_lwe_list()
            .ciphertext_iter()
            .zip(gsw.as_lwe_list().ciphertext_iter())
            .zip(
                seeded_decryption
                    .plaintext_iter_mut()
                    .zip(control_decryption.plaintext_iter_mut()),
            )
        {
            // decrypts
            sk.decrypt_lwe(decoded_seeded, &seeded_lwe);
            sk.decrypt_lwe(decoded_control, &control_lwe);

            // test

            decoded_seeded.0 >>= T::BITS - 22;
            if decoded_seeded.0 & T::ONE == T::ONE {
                decoded_seeded.0 += T::ONE;
            }
            decoded_seeded.0 >>= 1;
            decoded_seeded.0 %= T::ONE << 21;

            decoded_control.0 >>= T::BITS - 22;
            if decoded_control.0 & T::ONE == T::ONE {
                decoded_control.0 += T::ONE;
            }
            decoded_control.0 >>= 1;
            decoded_control.0 %= T::ONE << 21;
            assert_eq!(decoded_seeded.0, decoded_control.0);
        }
    }
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

#[test]
pub fn test_seeded_gsw_u32() {
    test_seeded_gsw::<u32>()
}

#[test]
pub fn test_seeded_gsw_u64() {
    test_seeded_gsw::<u64>()
}
