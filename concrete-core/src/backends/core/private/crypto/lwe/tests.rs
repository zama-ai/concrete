use concrete_commons::dispersion::LogStandardDev;
use concrete_commons::key_kinds::BinaryKeyKind;

use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, LweDimension, PlaintextCount,
};
use concrete_npe as npe;

use crate::backends::core::private::crypto::encoding::PlaintextList;
use crate::backends::core::private::crypto::lwe::{LweKeyswitchKey, LweList};
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, UniformMsb};

use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{
    assert_delta_std_dev, assert_noise_distribution, random_ciphertext_count,
};

fn test_keyswitch<T: UnsignedTorus + RandomGenerable<UniformMsb>>() {
    //! create a KSK and key switch some LWE samples
    //! warning: not a randomized test for the parameters
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // fix a set of parameters
    let n_bit_msg = 8; // bit precision of the plaintext
    let nb_ct = random_ciphertext_count(100); // number of messages to encrypt
    let base_log = DecompositionBaseLog(3); // a parameter of the gadget matrix
    let level_count = DecompositionLevelCount(8); // a parameter of the gadget matrix
    let messages = PlaintextList::from_tensor(
        random_generator.random_uniform_n_msb_tensor(nb_ct.0, n_bit_msg),
    );
    // the set of messages to encrypt
    let std_input = LogStandardDev::from_log_standard_dev(-10.); // standard deviation of the
                                                                 // encrypted messages to KS
    let std_ksk = LogStandardDev::from_log_standard_dev(-25.); // standard deviation of the ksk

    // set parameters related to the after (stands for 'after the KS')
    let dimension_after = LweDimension(600);
    let sk_after = LweSecretKey::generate_binary(dimension_after, &mut secret_generator);

    // set parameters related to the before (stands for 'before the KS')
    let dimension_before = LweDimension(1024);
    let sk_before = LweSecretKey::generate_binary(dimension_before, &mut secret_generator);

    // create the before ciphertexts and the after ciphertexts
    let mut ciphertexts_before = LweList::allocate(T::ZERO, dimension_before.to_lwe_size(), nb_ct);
    let mut ciphertexts_after = LweList::allocate(T::ZERO, dimension_after.to_lwe_size(), nb_ct);

    // key switching key generation
    let mut ksk = LweKeyswitchKey::allocate(
        T::ZERO,
        level_count,
        base_log,
        dimension_before,
        dimension_after,
    );
    ksk.fill_with_keyswitch_key(&sk_before, &sk_after, std_ksk, &mut encryption_generator);

    // encrypts with the before key our messages
    sk_before.encrypt_lwe_list(
        &mut ciphertexts_before,
        &messages,
        std_input,
        &mut encryption_generator,
    );

    // key switch before -> after
    ksk.keyswitch_list(&mut ciphertexts_after, &ciphertexts_before);

    // decryption with the after key
    let mut dec_messages = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    sk_after.decrypt_lwe_list(&mut dec_messages, &ciphertexts_after);

    // calls the NPE to find out the amount of noise after KS
    let output_variance = npe::estimate_keyswitch_noise_lwe_to_glwe_with_constant_terms::<
        T,
        _,
        _,
        BinaryKeyKind,
    >(dimension_before, std_input, std_ksk, base_log, level_count);

    if nb_ct.0 < 7 {
        // assert the difference between the original messages and the decrypted messages
        assert_delta_std_dev(&messages, &dec_messages, output_variance);
    } else {
        assert_noise_distribution(&messages, &dec_messages, output_variance);
    }
}

#[test]
fn test_keyswitch_u32() {
    test_keyswitch::<u32>();
}

#[test]
fn test_keyswitch_u64() {
    test_keyswitch::<u64>();
}
