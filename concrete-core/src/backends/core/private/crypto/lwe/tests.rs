use concrete_commons::dispersion::{LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::{CastFrom, Numeric, SignedInteger};
use concrete_commons::parameters::{
    CiphertextCount, CleartextCount, DecompositionBaseLog, DecompositionLevelCount, LweDimension,
    PlaintextCount,
};
use concrete_npe as npe;

use crate::backends::core::private::crypto::encoding::{
    Cleartext, CleartextList, Plaintext, PlaintextList,
};
use crate::backends::core::private::crypto::lwe::{LweCiphertext, LweKeyswitchKey, LweList};
use crate::backends::core::private::crypto::secret::generators::{
    EncryptionRandomGenerator, SecretRandomGenerator,
};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::random::{RandomGenerable, RandomGenerator, UniformMsb};
use crate::backends::core::private::math::tensor::{AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::test_tools::{
    assert_delta_std_dev, assert_noise_distribution, random_ciphertext_count, random_lwe_dimension,
    random_uint_between,
};

use super::{LweSeededCiphertext, LweSeededKeyswitchKey, LweSeededList};

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

fn test_multisum_npe<T>()
where
    T: UnsignedTorus + RandomGenerable<UniformMsb> + CastFrom<usize>,
{
    //! encrypts messages, does a multisum and decrypts the result
    //! warning: std_dev is not randomized
    let mut new_msg = Tensor::allocate(T::ZERO, 100);
    let mut msg = Tensor::allocate(T::ZERO, 100);
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // generate random settings
    let nb_ct = random_ciphertext_count(100);
    let dimension = random_lwe_dimension(1000);
    let std = LogStandardDev::from_log_standard_dev(-25.);

    // generate random weights
    let mut weights = CleartextList::allocate(T::ZERO, CleartextCount(nb_ct.0));
    let mut s_weights = CleartextList::allocate(T::Signed::ZERO, CleartextCount(nb_ct.0));
    for (w, sw) in weights
        .as_mut_tensor()
        .iter_mut()
        .zip(s_weights.as_mut_tensor().iter_mut())
    {
        *sw = random_uint_between::<T>(T::ZERO..T::cast_from(512)).into_signed()
            - T::cast_from(256).into_signed();
        *w = sw.into_unsigned();
    }
    let bias = Plaintext(random_uint_between::<T>(T::ZERO..T::cast_from(1024)));

    let n_tests = 10;
    for i in 0..n_tests {
        // generate the secret key
        let sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

        // generate random messages
        let mut messages = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
        random_generator.fill_tensor_with_random_uniform(&mut messages);

        // generate trivial encryptions for the witness
        let witness = LweList::new_trivial_encryption(dimension.to_lwe_size(), &messages);

        // generate ciphertexts with the secret key
        let mut ciphertext = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
        sk.encrypt_lwe_list(&mut ciphertext, &messages, std, &mut encryption_generator);

        // allocation for the results
        let mut ct_res = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());
        let mut ct_res_witness = LweCiphertext::allocate(T::ZERO, dimension.to_lwe_size());

        // computation of the multisums
        ct_res.fill_with_multisum_with_bias(&ciphertext, &weights, &bias);
        ct_res_witness.fill_with_multisum_with_bias(&witness, &weights, &bias);

        // decryption
        let mut output = Plaintext(T::ZERO);
        sk.decrypt_lwe(&mut output, &ct_res);

        new_msg.set_element(i, output.0);
        msg.set_element(i, ct_res_witness.get_body().0);
    }

    // noise prediction
    let mut weights: Vec<T> = vec![T::ZERO; s_weights.as_tensor().len()];
    for (w, sw) in weights.iter_mut().zip(s_weights.as_tensor().iter()) {
        *w = sw.into_unsigned();
    }
    let output_variance: Variance = npe::estimate_weighted_sum_noise::<T, _>(
        &vec![Variance(f64::powi(std.0, 2)); nb_ct.0],
        &weights,
    );

    if n_tests < 7 {
        assert_delta_std_dev(&new_msg, &msg, output_variance);
    } else {
        assert_noise_distribution(&msg, &new_msg, output_variance);
    }
}

#[test]
fn test_multisum_u32() {
    test_multisum_npe::<u32>();
}

#[test]
fn test_multisum_u64() {
    test_multisum_npe::<u64>();
}

fn test_scalar_mul<T>()
where
    T: UnsignedTorus + RandomGenerable<UniformMsb> + CastFrom<usize>,
{
    //! encrypts a bunch of messages, performs a scalar multiplication and decrypts them
    //! the settings are not randomized
    // settings
    let n_tests = 10;
    let nb_ct = CiphertextCount(n_tests);
    let dimension = LweDimension(600);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encryption_generator = EncryptionRandomGenerator::new(None);

    // generate the secret key
    let sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

    // generate random messages
    let mut messages = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    random_generator.fill_tensor_with_random_uniform(&mut messages);

    // encryption
    let mut ciphertexts = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
    sk.encrypt_lwe_list(
        &mut ciphertexts,
        &messages,
        std_dev,
        &mut encryption_generator,
    );

    // generate a random signed weight vector represented as Torus elements
    let weight = Cleartext(
        (random_uint_between::<T>(T::ZERO..T::cast_from(1024)).into_signed()
            - T::cast_from(512).into_signed())
        .into_unsigned(),
    );

    // scalar mul
    let mut ciphertext_sm = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
    ciphertext_sm
        .ciphertext_iter_mut()
        .zip(ciphertexts.ciphertext_iter())
        .for_each(|(mut out, inp)| out.fill_with_scalar_mul(&inp, &weight));

    // compute on cleartexts the multiplication
    let mut messages_mul = Tensor::allocate(T::ZERO, nb_ct.0);
    messages_mul.fill_with_one(messages.as_tensor(), |m| m.wrapping_mul(weight.0));

    // decryption
    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    sk.decrypt_lwe_list(&mut decryptions, &ciphertext_sm);

    // test
    let output_variance: Variance = npe::estimate_integer_plaintext_multiplication_noise::<T, _>(
        Variance(f64::powi(std_dev.0, 2)),
        weight.0,
    );
    if nb_ct.0 < 7 {
        // assert the difference between the original messages and the decrypted messages
        assert_delta_std_dev(&messages_mul, &decryptions, output_variance);
    } else {
        assert_noise_distribution(&messages_mul, &decryptions, output_variance);
    }
}

#[test]
fn test_scalar_mul_u32() {
    test_scalar_mul::<u32>();
}

#[test]
fn test_scalar_mul_u64() {
    test_scalar_mul::<u64>();
}

fn test_scalar_mul_random<T>()
where
    T: UnsignedTorus + RandomGenerable<UniformMsb> + CastFrom<usize>,
{
    //! encrypts a bunch of messages, performs a scalar multiplication and decrypts them
    //! warning: std_dev is not randomized
    //! only assert with assert_delta_std_dev

    // settings
    let nb_ct = random_ciphertext_count(100);
    let dimension = random_lwe_dimension(1000);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);
    let mut random_generator = RandomGenerator::new(None);
    let mut secret_generator = SecretRandomGenerator::new(None);
    let mut encrytion_generator = EncryptionRandomGenerator::new(None);

    // generate the secret key
    let sk = LweSecretKey::generate_binary(dimension, &mut secret_generator);

    // generate random messages
    let mut messages = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    random_generator.fill_tensor_with_random_uniform(&mut messages);

    // encryption
    let mut ciphertexts = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
    sk.encrypt_lwe_list(
        &mut ciphertexts,
        &messages,
        std_dev,
        &mut encrytion_generator,
    );

    // generate a random signed weight vector as Torus elements
    let mut weights = CleartextList::allocate(T::ZERO, CleartextCount(nb_ct.0));
    for w in weights.as_mut_tensor().iter_mut() {
        let val = random_uint_between::<T>(T::ZERO..T::cast_from(1024)).into_signed()
            - T::cast_from(512).into_signed();
        *w = val.into_unsigned();
    }

    // scalar mul
    let mut ciphertexts_out = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
    sk.encrypt_lwe_list(
        &mut ciphertexts,
        &messages,
        std_dev,
        &mut encrytion_generator,
    );

    for (mut out, (inp, w)) in ciphertexts_out
        .ciphertext_iter_mut()
        .zip(ciphertexts.ciphertext_iter().zip(weights.cleartext_iter()))
    {
        out.fill_with_scalar_mul(&inp, w);
    }

    // compute on cleartexts the multiplication
    let mut messages_mul = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    for (mm, (w_i, m)) in messages_mul
        .plaintext_iter_mut()
        .zip(weights.cleartext_iter().zip(messages.plaintext_iter()))
    {
        *mm = Plaintext(w_i.0.wrapping_mul(m.0));
    }

    // decryption
    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    sk.decrypt_lwe_list(&mut decryptions, &ciphertexts_out);

    // test
    for (mm, (d, w_i)) in messages_mul.sublist_iter(PlaintextCount(1)).zip(
        decryptions
            .sublist_iter(PlaintextCount(1))
            .zip(weights.cleartext_iter()),
    ) {
        // noise prediction work
        let output_variance: Variance = npe::estimate_integer_plaintext_multiplication_noise::<T, _>(
            Variance(f64::powi(std_dev.0, 2)),
            w_i.0,
        );
        assert_delta_std_dev(&mm, &d, output_variance);
    }
}

#[test]
fn test_scalar_mul_random_u32() {
    test_scalar_mul_random::<u32>()
}

#[test]
fn test_scalar_mul_random_u64() {
    test_scalar_mul_random::<u64>()
}

fn test_seeded_list_1<T>()
where
    T: UnsignedTorus,
    LweSeededList<Vec<T>>: AsRefTensor<Element = T>,
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

    // encryption
    let mut ciphertexts = LweSeededList::allocate(T::ZERO, dimension, nb_ct);
    sk.encrypt_seeded_lwe_list(&mut ciphertexts, &messages, std_dev);

    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    for (mut decryption, encryption) in decryptions
        .plaintext_iter_mut()
        .zip(ciphertexts.ciphertext_iter())
    {
        let mut expanded = LweCiphertext::allocate(T::ZERO, encryption.lwe_size());
        encryption.expand_into(&mut expanded);
        sk.decrypt_lwe(&mut decryption, &expanded);
    }

    // make sure that after decryption we recover the original plaintext
    if nb_ct.0 < 7 {
        assert_delta_std_dev(&messages, &decryptions, std_dev);
    } else {
        assert_noise_distribution(&messages, &decryptions, std_dev);
    }
}

#[test]
fn test_seeded_list_1_u32() {
    test_seeded_list_1::<u32>()
}

#[test]
fn test_seeded_list_1_u64() {
    test_seeded_list_1::<u64>()
}

fn test_seeded_list_2<T>()
where
    T: UnsignedTorus,
    LweSeededList<Vec<T>>: AsRefTensor<Element = T>,
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

    // encryption
    let mut ciphertexts = LweSeededList::allocate(T::ZERO, dimension, nb_ct);
    sk.encrypt_seeded_lwe_list(&mut ciphertexts, &messages, std_dev);

    let mut expanded = LweList::allocate(T::ZERO, dimension.to_lwe_size(), nb_ct);
    ciphertexts.expand_into(&mut expanded);

    let mut decryptions = PlaintextList::allocate(T::ZERO, PlaintextCount(nb_ct.0));
    sk.decrypt_lwe_list(&mut decryptions, &expanded);

    // make sure that after decryption we recover the original plaintext
    if nb_ct.0 < 7 {
        assert_delta_std_dev(&messages, &decryptions, std_dev);
    } else {
        assert_noise_distribution(&messages, &decryptions, std_dev);
    }
}

#[test]
fn test_seeded_list_2_u32() {
    test_seeded_list_2::<u32>()
}

#[test]
fn test_seeded_list_2_u64() {
    test_seeded_list_2::<u64>()
}

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

    for (decryption, message) in decryptions
        .plaintext_iter_mut()
        .zip(messages.plaintext_iter())
    {
        // encryption
        let mut ciphertext = LweSeededCiphertext::allocate(T::ZERO, dimension);
        sk.encrypt_seeded_lwe(&mut ciphertext, message, std_dev);

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

fn test_seeded_keyswitch_key<T>()
where
    T: UnsignedTorus,
{
    // settings
    let nb_test = random_ciphertext_count(10);
    let dimension_in = random_lwe_dimension(1000);
    let dimension_out = random_lwe_dimension(1000);
    let std_dev = LogStandardDev::from_log_standard_dev(-15.);
    let decomp_level = DecompositionLevelCount(3);
    let decomp_base_log = DecompositionBaseLog(7);

    let mut secret_generator = SecretRandomGenerator::new(None);

    for _ in 0..nb_test.0 {
        // generate the secret keys
        let sk_in = LweSecretKey::generate_binary(dimension_in, &mut secret_generator);
        let sk_out = LweSecretKey::generate_binary(dimension_out, &mut secret_generator);

        let mut seeded_ksk = LweSeededKeyswitchKey::allocate(
            T::ZERO,
            decomp_level,
            decomp_base_log,
            dimension_in,
            dimension_out,
        );

        let mut encryption_generator = EncryptionRandomGenerator::new(Some(seeded_ksk.get_seed()));
        let mut ksk = LweKeyswitchKey::allocate(
            T::ZERO,
            decomp_level,
            decomp_base_log,
            dimension_in,
            dimension_out,
        );

        seeded_ksk.fill_with_seeded_keyswitch_key(&sk_in, &sk_out, std_dev);
        ksk.fill_with_keyswitch_key(&sk_in, &sk_out, std_dev, &mut encryption_generator);
        assert_eq!(
            seeded_ksk.as_tensor().as_slice(),
            ksk.as_tensor().as_slice()
        );
    }
}

#[test]
fn test_seeded_keyswitch_key_u32() {
    test_seeded_ciphertext::<u32>()
}

#[test]
fn test_seeded_keyswitch_key_u64() {
    test_seeded_ciphertext::<u64>()
}
