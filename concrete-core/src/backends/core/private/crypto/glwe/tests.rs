use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
use crate::backends::core::private::crypto::secret::GlweSecretKey;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools;
use crate::backends::core::private::test_tools::assert_delta_std_dev;
use concrete_commons::dispersion::LogStandardDev;

use super::{GlweCiphertext, GlweSeededCiphertext};

fn test_seeded_glwe<T: UnsignedTorus>() {
    // random settings
    let dimension = test_tools::random_glwe_dimension(5);
    let polynomial_size = test_tools::random_polynomial_size(200);
    let noise_parameter = LogStandardDev::from_log_standard_dev(-20.);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut seed_generator = RandomGenerator::new(None);

    // generates a secret key
    let sk = GlweSecretKey::generate_binary(dimension, polynomial_size, &mut secret_generator);

    // generates random plaintexts
    let plaintext_vector =
        PlaintextList::from_tensor(secret_generator.random_uniform_tensor(polynomial_size.0));

    // encrypts
    let seed = seed_generator.generate_seed();
    let mut ciphertext = GlweSeededCiphertext::allocate(T::ZERO, polynomial_size, dimension);
    sk.encrypt_seeded_glwe(&mut ciphertext, &plaintext_vector, noise_parameter, seed);

    // expands
    let mut ciphertext_expanded =
        GlweCiphertext::allocate(T::ZERO, polynomial_size, dimension.to_glwe_size());
    ciphertext.expand_into(&mut ciphertext_expanded);

    // decrypts
    let mut decryptions =
        PlaintextList::from_tensor(secret_generator.random_uniform_tensor(polynomial_size.0));
    sk.decrypt_glwe(&mut decryptions, &ciphertext_expanded);

    // test
    assert_delta_std_dev(&plaintext_vector, &decryptions, noise_parameter);
}

#[test]
fn test_seeded_glwe_u32() {
    test_seeded_glwe::<u32>();
}

#[test]
fn test_seeded_glwe_u64() {
    test_seeded_glwe::<u64>();
}
