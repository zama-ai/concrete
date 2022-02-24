use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::parameters::PlaintextCount;

use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{
    assert_delta_std_dev, assert_noise_distribution, random_ciphertext_count, random_lwe_dimension,
};

use super::LweSeededCiphertext;

fn test_seeded_ciphertext<T>()
where
    T: UnsignedTorus,
{
    // settings
    let nb_ct = random_ciphertext_count(100);
    let dimension = random_lwe_dimension(1000);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);

    // generate the secret key
    let mut generator = SecretRandomGenerator::new(None);
    let sk = LweSecretKey::generate_binary(dimension, &mut generator);

    // generate random messages
    let messages = PlaintextList::from_tensor(generator.random_uniform_tensor(nb_ct.0));
    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));

    let mut seed_generator = RandomGenerator::new(None);
    for (decryption, message) in decryptions
        .plaintext_iter_mut()
        .zip(messages.plaintext_iter())
    {
        // encryption
        let seed = seed_generator.generate_seed();
        let mut ciphertext = LweSeededCiphertext::allocate(T::ZERO, dimension);
        sk.encrypt_seeded_lwe(&mut ciphertext, message, std_dev, seed);

        let mut expanded = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        ciphertext.expand_into(&mut expanded);

        sk.decrypt_lwe(decryption, &expanded);
    }

    // make sure that after decryption we recover the original plaintext
    if nb_ct.0 < 7 {
        assert_delta_std_dev(&messages, &decryptions, std_dev);
    } else {
        assert_noise_distribution(&messages, &decryptions, std_dev);
    }
}

#[test]
fn test_seeded_ciphertext_u32() {
    test_seeded_ciphertext::<u32>()
}

#[test]
fn test_seeded_ciphertext_u64() {
    test_seeded_ciphertext::<u64>()
}
