#![allow(clippy::modulo_one)]

use itertools::izip;

#[test]
fn test_encode_encrypt_x_copy_in_nth_nth_inplace_x_decrypt() {
    // random number of messages
    let nb_messages_1: usize = random_index!(20) + 1;
    let nb_messages_2: usize = random_index!(20) + 1;

    // random settings for the first encoder and some messages
    let (min1, max1) = generate_random_interval!();
    let (precision1, padding1) = generate_precision_padding!(8, 8);
    let encoder_1 = crate::Encoder::new(min1, max1, precision1, padding1).unwrap();
    let messages_1: Vec<f64> = random_messages!(min1, max1, nb_messages_1);

    // random settings for the second encoder and some random messages
    let (min2, max2) = generate_random_interval!();
    let (precision2, padding2) = generate_precision_padding!(8, 8);
    let encoder_2 = crate::Encoder::new(min2, max2, precision2, padding2).unwrap();
    let messages_2: Vec<f64> = random_messages!(min2, max2, nb_messages_2);

    // generate a secret key
    let dimension: usize = random_index!(1024) + 1;
    let log_std_dev: i32 = -(random_index!(40) as i32 + 20);
    let params = crate::LWEParams::new(dimension, log_std_dev);
    let sk = crate::LWESecretKey::new(&params);

    // encode and encrypt
    let mut ct1 = crate::VectorLWE::encode_encrypt(&sk, &messages_1, &encoder_1).unwrap();
    let ct2 = crate::VectorLWE::encode_encrypt(&sk, &messages_2, &encoder_2).unwrap();

    // random indexes
    let index1: usize = random_index!(nb_messages_1);
    let index2: usize = random_index!(nb_messages_2);

    // copy
    ct1.copy_in_nth_nth_inplace(index1, &ct2, index2).unwrap();

    // decryption
    let decryptions = ct1.decrypt_decode_round(&sk).unwrap();

    // test
    let mut cpt: usize = 0;
    for (i, m, d, enc) in izip!(
        0..nb_messages_1,
        messages_1.iter(),
        decryptions.iter(),
        ct1.encoders.iter()
    ) {
        if i != index1 {
            assert_eq_granularity!(m, d, enc);
            assert_eq!(precision1, enc.nb_bit_precision);
        } else {
            assert_eq_granularity!(messages_2[index2], d, enc);
            assert_eq!(precision2, enc.nb_bit_precision);
        }
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages_1);
}

#[test]
fn test_encrypt_x_extract_nth_x_decrypt() {
    let nb_messages: usize = random_index!(20) + 1;

    // random settings for the first encoder and some messages
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();
    let messages: Vec<f64> = random_messages!(min, max, nb_messages);

    // generate a secret key
    let dimension: usize = random_index!(1024) + 1;
    let log_std_dev: i32 = -(random_index!(40) as i32 + 20);
    let params = crate::LWEParams::new(dimension, log_std_dev);
    let sk = crate::LWESecretKey::new(&params);

    // encode and encrypt
    let pt = encoder.encode(&messages).unwrap();
    let ct = crate::VectorLWE::encrypt(&sk, &pt).unwrap();

    // random indexes
    let index: usize = random_index!(nb_messages);
    let ct_extracted = ct.extract_nth(index).unwrap();

    // decryption
    let decryptions = ct_extracted.decrypt_decode_round(&sk).unwrap();

    // test
    assert_eq_granularity!(messages[index], decryptions[0], ct_extracted.encoders[0]);
    assert_eq!(precision, ct_extracted.encoders[0].nb_bit_precision);
}

#[test]
fn test_encrypt_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let nb_messages: usize = random_index!(30) + 10;

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // a list of messages
    let messages: Vec<f64> = random_messages!(min, max, nb_messages);

    // encode and encrypt
    let plaintext = crate::Plaintext::encode(&messages, &encoder).unwrap();
    let ciphertext = crate::VectorLWE::encrypt(&secret_key, &plaintext).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // test
    let mut cpt: usize = 0;
    for (m, d, e) in izip!(
        messages.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        assert_eq_granularity!(m, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encrypt_x_add_constant_static_encoder_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let nb_messages: usize = random_index!(30) + 10;

    // encoder
    let encoder = crate::Encoder::new(min - 10., max + 10., precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let messages_1: Vec<f64> = random_messages!(min, max, nb_messages);
    let messages_2: Vec<f64> = random_messages!(-10., 10., nb_messages);

    // encode and encrypt
    let plaintext_1 = crate::Plaintext::encode(&messages_1, &encoder).unwrap();
    let mut ciphertext = crate::VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();

    // addition between ciphertext and messages_2
    ciphertext
        .add_constant_static_encoder_inplace(&messages_2)
        .unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // test
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages_1.iter(),
        messages_2.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        assert_eq_granularity!(m1 + m2, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_new_encode_encrypt_x_add_constant_dynamic_encoder_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, max2) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let nb_messages: usize = random_index!(30) + 10;

    // encoder
    let encoder = crate::Encoder::new(min1, max1, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let messages_1: Vec<f64> = random_messages!(min1, max1, nb_messages);
    let messages_2: Vec<f64> = random_messages!(min2, max2, nb_messages);

    // encode and encrypt
    let plaintext_1 = crate::Plaintext::encode(&messages_1, &encoder).unwrap();
    let mut ciphertext = crate::VectorLWE::encrypt(&secret_key, &plaintext_1).unwrap();

    // addition between ciphertext and messages_2
    ciphertext
        .add_constant_dynamic_encoder_inplace(&messages_2)
        .unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // test
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages_1.iter(),
        messages_2.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        assert_eq_granularity!(m1 + m2, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_opposite_nth_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let nb_messages: usize = random_index!(30) + 10;
    let index: usize = random_index!(nb_messages);

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // a list of messages
    let messages: Vec<f64> = random_messages!(min, max, nb_messages);

    println!(
        "padding {} precision {} nb {} index {} m {} encoder {}",
        padding, precision, nb_messages, index, messages[index], encoder,
    );

    // encode and encrypt
    let mut ciphertext =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages, &encoder).unwrap();

    // compute the opposite of the second ciphertext
    ciphertext.opposite_nth_inplace(index).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // test
    let mut cpt: usize = 0;
    for (i, m, d, enc) in izip!(
        0..nb_messages,
        messages.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        if i != index {
            assert_eq_granularity!(m, d, enc);
        } else {
            assert_eq_granularity!(-m, d, enc);
        }
        assert_eq!(precision, enc.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_add_centered_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (min2, _) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let nb_messages: usize = random_index!(30) + 10;

    // encoders
    let encoder1 = crate::Encoder::new(min, max, precision, padding).unwrap();
    let max2 = min2 + max - min;
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let delta = (max - min) / 4. + encoder1.get_granularity() / 2.;
    let messages1: Vec<f64> =
        random_messages!(min + delta, min + encoder1.delta - delta, nb_messages);
    let messages2: Vec<f64> =
        random_messages!(min2 + delta, min2 + encoder2.delta - delta, nb_messages);

    // encode and encrypt
    let mut ciphertext1 =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages1, &encoder1).unwrap();
    let ciphertext2 = crate::VectorLWE::encode_encrypt(&secret_key, &messages2, &encoder2).unwrap();

    // addition between ciphertext1 and ciphertext2
    ciphertext1.add_centered_inplace(&ciphertext2).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

    // check the precision loss related to the encryption
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext1.encoders.iter()
    ) {
        assert_eq_granularity!(m1 + m2, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_add_with_padding_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(8, 3);
    padding += 1; // at least one bit
    let nb_messages: usize = random_index!(30) + 10;

    // encoders
    let encoder1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let messages1: Vec<f64> = random_messages!(min1, max1, nb_messages);
    let messages2: Vec<f64> = random_messages!(min2, max2, nb_messages);

    // encode and encrypt
    let mut ciphertext1 =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages1, &encoder1).unwrap();
    let ciphertext2 = crate::VectorLWE::encode_encrypt(&secret_key, &messages2, &encoder2).unwrap();

    // addition between ciphertext and messages_2
    ciphertext1.add_with_padding_inplace(&ciphertext2).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

    // check the precision loss related to the encryption
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext1.encoders.iter()
    ) {
        assert_eq_granularity!(m1 + m2, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_sub_with_padding_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(8, 3);
    padding += 1; // at least one bit
    let nb_messages: usize = random_index!(30) + 10;

    // encoders
    let encoder1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let messages1: Vec<f64> = random_messages!(min1, max1, nb_messages);
    let messages2: Vec<f64> = random_messages!(min2, max2, nb_messages);

    // encode and encrypt
    let mut ciphertext1 =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages1, &encoder1).unwrap();
    let ciphertext2 = crate::VectorLWE::encode_encrypt(&secret_key, &messages2, &encoder2).unwrap();

    // subtraction between ciphertext and messages_2
    ciphertext1.sub_with_padding_inplace(&ciphertext2).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

    // check the precision loss related to the encryption
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext1.encoders.iter()
    ) {
        assert_eq_granularity!(m1 - m2, d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_mul_constant_static_encoder_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_centered_interval!();
    let (precision, padding) = generate_precision_padding!(6, 2);
    let nb_messages: usize = random_index!(30) + 10;
    let b = min.abs().min(max.abs()) / 20.;

    // encoders
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // two lists of messages
    let messages1: Vec<f64> = random_messages!(-b, b, nb_messages);
    let messages2_float: Vec<f64> = random_messages!(-b, b, nb_messages);
    let mut messages2: Vec<i32> = vec![0; nb_messages];
    for (m, f) in izip!(messages2.iter_mut(), messages2_float.iter()) {
        *m = *f as i32;
    }

    // encode and encrypt
    let mut ciphertext =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages1, &encoder).unwrap();

    // multiplication between ciphertext and messages2
    ciphertext
        .mul_constant_static_encoder_inplace(&messages2)
        .unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // check the precision loss related to the encryption
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        assert_eq_granularity!(*m1 * (*m2 as f64), d, e);
        assert_eq!(precision, e.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_mul_constant_with_padding_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_centered_interval!();
    let precision: usize = random_index!(5) + 3;
    let padding = random_index!(3) + precision;
    let nb_messages: usize = random_index!(30) + 10;
    let nb_bit_padding_mult = precision;
    let b = (random_index!(300) + 3) as f64;

    // encoders
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_2048);

    // two lists of messages
    let messages1: Vec<f64> = random_messages!(min, max, nb_messages);
    let messages2: Vec<f64> = random_messages!(-b, b, nb_messages);

    // encode and encrypt
    let mut ciphertext =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages1, &encoder).unwrap();

    // multiplication between ciphertext and messages2
    ciphertext
        .mul_constant_with_padding_inplace(&messages2, b, nb_bit_padding_mult)
        .unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext.decrypt_decode_round(&secret_key).unwrap();

    // check the precision loss related to the encryption
    let mut cpt: usize = 0;
    for (m1, m2, d, e) in izip!(
        messages1.iter(),
        messages2.iter(),
        decryptions.iter(),
        ciphertext.encoders.iter()
    ) {
        assert_eq_granularity!(m1 * m2, d, e);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_keyswitch_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(4, 1);
    let nb_messages: usize = random_index!(30) + 10;
    let base_log: usize = 9;
    let level: usize = 7;

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate two secret keys
    let secret_key_before = crate::LWESecretKey::new(&crate::LWE128_1024);
    let secret_key_after = crate::LWESecretKey::new(&crate::LWE128_1024);

    // generate the key switching key
    let ksk = crate::LWEKSK::new(&secret_key_before, &secret_key_after, base_log, level);

    // a list of messages that we encrypt
    let messages: Vec<f64> = random_messages!(min, max, nb_messages);
    let ciphertext_before =
        crate::VectorLWE::encode_encrypt(&secret_key_before, &messages, &encoder).unwrap();

    // key switch
    let ciphertext_after = ciphertext_before.keyswitch(&ksk).unwrap();

    // decryption
    let decryptions: Vec<f64> = ciphertext_after
        .decrypt_decode_round(&secret_key_after)
        .unwrap();

    let mut cpt: usize = 0;
    for (m, d, e) in izip!(
        messages.iter(),
        decryptions.iter(),
        ciphertext_after.encoders.iter()
    ) {
        assert_eq_granularity!(*m, d, e);
        println!("{} {}", m, d);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_x_bootstrap_nth_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let padding: usize = 1;
    let precision: usize = random_index!(3) + 1;
    let base_log: usize = random_index!(3) + 7;
    let level: usize = random_index!(1) + 3;
    let nb_messages: usize = random_index!(30) + 10;

    for _i in 0..1 {
        // encoders
        let encoder_input = crate::Encoder::new(min, max, precision, padding).unwrap();

        // secret keys
        let rlwe_secret_key = crate::RLWESecretKey::new(&crate::RLWE128_1024_1);
        let secret_key_input = crate::LWESecretKey::new(&crate::LWE128_630);
        let secret_key_output = rlwe_secret_key.to_lwe_secret_key();

        // bootstrapping key
        let bootstrapping_key =
            crate::LWEBSK::new(&secret_key_input, &rlwe_secret_key, base_log, level);

        // messages
        let message: Vec<f64> = random_messages!(min, max, nb_messages);

        // encode and encrypt
        let ciphertext_input =
            crate::VectorLWE::encode_encrypt(&secret_key_input, &message, &encoder_input).unwrap();

        let mut cpt: usize = 0;
        for (index, item) in message.iter().enumerate() {
            // bootstrap
            let ciphertext_output = ciphertext_input
                .bootstrap_nth(&bootstrapping_key, index)
                .unwrap();

            // decrypt
            let decryption2 = ciphertext_output
                .decrypt_decode_round(&secret_key_output)
                .unwrap();
            assert_eq_granularity!(
                item,
                decryption2[0],
                ciphertext_output.encoders[0]
            );
            cpt += 1;
        }
        assert_eq!(cpt, nb_messages);
    }
}

#[test]
fn test_encode_encrypt_x_mul_from_bootstrap_nth_nth_x_decrypt() {
    let nb_messages: usize = 1;

    // random settings for the first encoder and some messages
    let (min1, max1) = generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, 5, 2).unwrap();
    let messages_1: Vec<f64> = random_messages!(min1, max1, nb_messages);

    // random settings for the second encoder and some messages
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1;
    let encoder_2 = crate::Encoder::new(min2, max2, 5, 2).unwrap();
    let messages_2: Vec<f64> = random_messages!(min2, max2, nb_messages);
    println!("encoder1 {}", encoder_1);
    println!("encoder2 {}", encoder_2);

    // generate a secret key
    let rlwe_secret_key = crate::RLWESecretKey::new(&crate::RLWE128_1024_1);
    let secret_key_input = crate::LWESecretKey::new(&crate::LWE128_630);
    let secret_key_output = rlwe_secret_key.to_lwe_secret_key();

    // bootstrapping key
    let bsk = crate::LWEBSK::new(&secret_key_input, &rlwe_secret_key, 5, 3);

    // encode and encrypt
    let ciphertext_1 =
        crate::VectorLWE::encode_encrypt(&secret_key_input, &messages_1, &encoder_1).unwrap();
    let ciphertext_2 =
        crate::VectorLWE::encode_encrypt(&secret_key_input, &messages_2, &encoder_2).unwrap();

    // random indexes
    let index1: usize = random_index!(nb_messages);
    let index2: usize = random_index!(nb_messages);

    // multiplication
    let ciphertext_res = ciphertext_1
        .mul_from_bootstrap_nth(&ciphertext_2, &bsk, index1, index2)
        .unwrap();

    // decrypt
    let decryption = ciphertext_res
        .decrypt_decode_round(&secret_key_output)
        .unwrap();

    // test
    assert_eq_granularity!(
        messages_1[index1] * messages_2[index2],
        decryption[0],
        ciphertext_res.encoders[0]
    );
}

#[test]
fn test_encode_encrypt_x_add_with_new_min_inplace_x_decrypt() {
    // random number of messages
    let nb_messages: usize = random_index!(20) + 1;
    let (precision, padding) = generate_precision_padding!(8, 8);

    // random settings for the first encoder and some messages
    let (min1, max1) = generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();
    let messages_1: Vec<f64> =
        random_messages!(min1 + encoder_1.get_size() / 2., max1, nb_messages);

    // random settings for the second encoder and some random messages
    let (min2, _max2) = generate_random_interval!();
    let encoder_2 =
        crate::Encoder::new(min2, min2 + encoder_1.get_size(), precision, padding).unwrap();
    let messages_2: Vec<f64> =
        random_messages!(min2, min2 + encoder_1.get_size() / 2., nb_messages);

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // encode and encrypt
    let mut ciphertext_1 =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages_1, &encoder_1).unwrap();
    let ciphertext_2 =
        crate::VectorLWE::encode_encrypt(&secret_key, &messages_2, &encoder_2).unwrap();

    // new_min
    let new_min: Vec<f64> = vec![min1 + min2 + encoder_1.get_size() / 2.; messages_1.len()];

    // addition between ciphertext_1 and ciphertext_2
    ciphertext_1
        .add_with_new_min_inplace(&ciphertext_2, &new_min)
        .unwrap();

    // decryption
    let decryptions = ciphertext_1.decrypt_decode_round(&secret_key).unwrap();

    // test
    let mut cpt: usize = 0;
    for (m1, m2, d, enc) in izip!(
        messages_1.iter(),
        messages_2.iter(),
        decryptions.iter(),
        ciphertext_1.encoders.iter()
    ) {
        println!("m1 {} m2 {} m1 + m2 {} dec {}", m1, m2, m1 + m2, d);
        assert_eq_granularity!(m1 + m2, d, enc);
        assert_eq!(precision, enc.nb_bit_precision);
        cpt += 1;
    }
    assert_eq!(cpt, nb_messages);
}

#[test]
fn test_encode_encrypt_several_encoders_x_sum_with_padding_x_decrypt_decode_round() {
    // random number of messages
    let nb_messages: usize = random_index!(20) + 1;
    let precision: usize = 5;
    let padding: usize = 5;

    // random settings for the first encoder
    let (min1, max1) = generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();

    // generate nb_messages encoders with the same length
    let mut encoders: Vec<crate::Encoder> = vec![encoder_1; nb_messages];
    for enc_in in encoders.iter_mut() {
        let (new_min, _) = generate_random_interval!();
        enc_in.o = new_min;
    }

    // generate messages
    let mut messages: Vec<f64> = vec![0.; nb_messages];
    for (m, enc_in) in izip!(messages.iter_mut(), encoders.iter()) {
        *m = random_message!(enc_in.o, enc_in.o + enc_in.get_size());
    }

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // encode and encrypt
    let ciphertexts =
        crate::VectorLWE::encode_encrypt_several_encoders(&secret_key, &messages, &encoders)
            .unwrap();

    // sum all the lwe together
    let ct_sum = ciphertexts.sum_with_padding().unwrap();

    // decryption
    let decryptions = ct_sum.decrypt_decode_round(&secret_key).unwrap();
    let d = decryptions[0];

    // test
    let mut clear_sum: f64 = 0.;
    for m in messages.iter() {
        clear_sum += m;
    }

    println!("sum {} dec {}", clear_sum, d);
    assert_eq_granularity!(clear_sum, d, ct_sum.encoders[0]);
    assert_eq!(precision, ct_sum.encoders[0].nb_bit_precision);
}

#[test]
fn test_encode_encrypt_several_encoders_x_sum_with_new_min_x_decrypt_decode_round() {
    // random number of messages
    let nb_messages: usize = 300; //random_index!(20) + 1;
    let precision: usize = 3; //5;
    let padding: usize = 0;

    // random settings for the first encoder
    let (min1, max1) = (0., 7.); //generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();

    // generate nb_messages encoders with the same length
    let mut encoders: Vec<crate::Encoder> = vec![encoder_1.clone(); nb_messages];
    for enc_in in encoders.iter_mut() {
        let (new_min, _) = generate_random_interval!();
        enc_in.o = f64::abs(f64::round(new_min));
    }

    // generate messages
    let mut messages: Vec<f64> = vec![0.; nb_messages];
    for (m, enc_in) in izip!(messages.iter_mut(), encoders.iter()) {
        *m = f64::round(random_message!(enc_in.o, enc_in.o + enc_in.get_size()));
        println!("message {}", *m);
    }

    // test
    let mut clear_sum: f64 = 0.;
    for m in messages.iter() {
        clear_sum += m;
    }

    // generate a new min
    let new_min = f64::round(clear_sum - random_message!(0., encoder_1.get_size()));

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    // encode and encrypt
    let ciphertexts =
        crate::VectorLWE::encode_encrypt_several_encoders(&secret_key, &messages, &encoders)
            .unwrap();

    // sum all the lwe together
    let ct_sum = ciphertexts.sum_with_new_min(new_min).unwrap();

    // decryption
    let decryptions = ct_sum.decrypt_decode_round(&secret_key).unwrap();
    let d = decryptions[0];

    println!("sum {} dec {}", clear_sum, d);
    assert_eq_granularity!(clear_sum, d, ct_sum.encoders[0]);
    assert_eq!(precision, ct_sum.encoders[0].nb_bit_precision);
}
