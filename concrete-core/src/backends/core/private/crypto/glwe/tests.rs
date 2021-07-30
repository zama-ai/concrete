use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::glwe::GlweList;
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::GlweSecretKey;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::AsRefTensor;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools;
use crate::backends::core::private::test_tools::assert_delta_std_dev;
use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::parameters::PlaintextCount;

use super::{GlweCiphertext, GlweSeededCiphertext, GlweSeededList};

fn test_glwe<T: UnsignedTorus>() {
    // random settings
    let nb_ct = test_tools::random_ciphertext_count(200);
    let dimension = test_tools::random_glwe_dimension(200);
    let polynomial_size = test_tools::random_polynomial_size(200);
    let noise_parameter = LogStandardDev::from_log_standard_dev(-20.);
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // generates a secret key
    let sk = GlweSecretKey::generate_binary(dimension, polynomial_size, &mut secret_generator);

    // generates random plaintexts
    let plaintexts = PlaintextList::from_tensor(
        random_generator.random_uniform_tensor(nb_ct.0 * polynomial_size.0),
    );

    // encrypts
    let mut ciphertext = GlweList::allocate(T::ZERO, polynomial_size, dimension, nb_ct);
    sk.encrypt_glwe_list(
        &mut ciphertext,
        &plaintexts,
        noise_parameter,
        &mut encryption_generator,
    );

    // decrypts
    let mut decryptions = PlaintextList::from_tensor(
        random_generator.random_uniform_tensor(nb_ct.0 * polynomial_size.0),
    );
    sk.decrypt_glwe_list(&mut decryptions, &ciphertext);

    // test
    assert_delta_std_dev(&plaintexts, &decryptions, noise_parameter);
}

#[test]
fn test_glwe_encrypt_decrypt_u32() {
    test_glwe::<u32>();
}

#[test]
fn test_glwe_encrypt_decrypt_u64() {
    test_glwe::<u64>();
}

fn test_seeded_glwe<T: UnsignedTorus>() {
    // random settings
    let dimension = test_tools::random_glwe_dimension(5);
    let polynomial_size = test_tools::random_polynomial_size(200);
    let noise_parameter = LogStandardDev::from_log_standard_dev(-20.);
    let mut generator = SecretRandomGenerator::new(None);

    // generates a secret key
    let sk = GlweSecretKey::generate_binary(dimension, polynomial_size, &mut generator);

    // generates random plaintexts
    let plaintext = PlaintextList::from_tensor(generator.random_uniform_tensor(polynomial_size.0));

    // encrypts
    let mut ciphertext = GlweSeededCiphertext::allocate(T::ZERO, polynomial_size, dimension);
    sk.encrypt_seeded_glwe(&mut ciphertext, &plaintext, noise_parameter.clone());

    // expands
    let mut ciphertext_expanded =
        GlweCiphertext::allocate(T::ZERO, polynomial_size, dimension.to_glwe_size());
    ciphertext.expand_into(&mut ciphertext_expanded);

    // decrypts
    let mut decryptions =
        PlaintextList::from_tensor(generator.random_uniform_tensor(polynomial_size.0));
    sk.decrypt_glwe(&mut decryptions, &ciphertext_expanded);

    // test
    assert_delta_std_dev(&plaintext, &decryptions, noise_parameter);
}

#[test]
fn test_seeded_glwe_u32() {
    test_seeded_glwe::<u32>();
}

#[test]
fn test_seeded_glwe_u64() {
    test_seeded_glwe::<u64>();
}

fn test_seeded_glwe_list_1<T>()
where
    T: UnsignedTorus,
    GlweSeededList<Vec<T>>: AsRefTensor<Element = T>,
{
    // settings
    let nb_ct = test_tools::random_ciphertext_count(200);
    let dimension = test_tools::random_glwe_dimension(5);
    let poly_size = test_tools::random_polynomial_size(200);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);

    // generate the secret key
    let mut generator = SecretRandomGenerator::new(None);
    let sk = GlweSecretKey::generate_binary(dimension, poly_size, &mut generator);

    // generate random messages
    let messages =
        PlaintextList::from_tensor(generator.random_uniform_tensor(poly_size.0 * nb_ct.0));

    // encryption
    let mut ciphertexts = GlweSeededList::allocate(T::ZERO, poly_size, dimension, nb_ct);
    sk.encrypt_seeded_glwe_list(&mut ciphertexts, &messages, std_dev);

    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0 * poly_size.0));
    for (mut decryption, encryption) in decryptions
        .sublist_iter_mut(PlaintextCount(poly_size.0))
        .zip(ciphertexts.ciphertext_iter())
    {
        let mut expanded = GlweCiphertext::allocate(
            T::ZERO,
            encryption.polynomial_size(),
            encryption.glwe_dimension.to_glwe_size(),
        );
        encryption.expand_into(&mut expanded);
        sk.decrypt_glwe(&mut decryption, &expanded);
    }

    // make sure that after decryption we recover the original plaintext
    assert_delta_std_dev(&messages, &decryptions, std_dev);
}

#[test]
fn test_seeded_glwe_list_1_u32() {
    test_seeded_glwe_list_1::<u32>()
}

#[test]
fn test_seeded_glwe_list_1_u64() {
    test_seeded_glwe_list_1::<u64>()
}

fn test_seeded_glwe_list_2<T>()
where
    T: UnsignedTorus,
    GlweSeededList<Vec<T>>: AsRefTensor<Element = T>,
{
    // settings
    let nb_ct = test_tools::random_ciphertext_count(200);
    let dimension = test_tools::random_glwe_dimension(5);
    let poly_size = test_tools::random_polynomial_size(200);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);

    // generate the secret key
    let mut generator = SecretRandomGenerator::new(None);
    let sk = GlweSecretKey::generate_binary(dimension, poly_size, &mut generator);

    // generate random messages
    let messages =
        PlaintextList::from_tensor(generator.random_uniform_tensor(poly_size.0 * nb_ct.0));

    // encryption
    let mut ciphertexts = GlweSeededList::allocate(T::ZERO, poly_size, dimension, nb_ct);
    sk.encrypt_seeded_glwe_list(&mut ciphertexts, &messages, std_dev);
    let mut expanded = GlweList::allocate(T::ZERO, poly_size, dimension, nb_ct);
    ciphertexts.expand_into(&mut expanded);

    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0 * poly_size.0));
    sk.decrypt_glwe_list(&mut decryptions, &expanded);

    // make sure that after decryption we recover the original plaintext
    assert_delta_std_dev(&messages, &decryptions, std_dev);
}

#[test]
fn test_seeded_glwe_list_2_u32() {
    test_seeded_glwe_list_2::<u32>()
}

#[test]
fn test_seeded_glwe_list_2_u64() {
    test_seeded_glwe_list_2::<u64>()
}
