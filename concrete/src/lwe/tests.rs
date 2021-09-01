#![allow(clippy::modulo_one)]

use crate::traits::GenericAdd;

#[test]
fn test_encode_encrypt_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // a message
        let message: f64 = random_message!(min, max);

        // encode and encrypt
        let ciphertext = crate::LWE::encode_encrypt(&secret_key, message, &encoder).unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message, decryption, ciphertext.encoder);
        assert_eq!(precision, ciphertext.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (min2, _) = generate_random_interval!();
    let (precision, mut padding) = generate_precision_padding!(8, 8);
    padding += 1;

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();
    let max2 = max - min + min2;
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message_1: f64 = random_message!(min, max);
        let message_2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let ciphertext_1 = crate::LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
        let ciphertext_2 = crate::LWE::encode_encrypt(&secret_key, message_2, &encoder2).unwrap();

        // addition between ciphertext and messages_2
        let ciphertext = ciphertext_1.add(&ciphertext_2).unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message_1 + message_2, decryption, ciphertext.encoder);
        assert_eq!(precision, ciphertext.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (min2, _) = generate_random_interval!();
    let (precision, mut padding) = generate_precision_padding!(8, 8);
    padding += 1;

    // encoders
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();
    let max2 = max - min + min2;
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two of message
        let message_1: f64 = random_message!(min, max);
        let message_2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let mut ciphertext_1 =
            crate::LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();
        let ciphertext_2 = crate::LWE::encode_encrypt(&secret_key, message_2, &encoder2).unwrap();

        // addition between ciphertext and messages_2
        ciphertext_1.add_inplace(&ciphertext_2).unwrap();

        // decryption
        let decryption: f64 = ciphertext_1.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message_1 + message_2, decryption, ciphertext_1.encoder);
        assert_eq!(precision, ciphertext_1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_constant_static_encoder_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // encoder
    let encoder = crate::Encoder::new(min - 10., max + 10., precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message_1: f64 = random_message!(min, max);
        let message_2: f64 = random_message!(-10., 10.);

        // encode and encrypt
        let mut ciphertext = crate::LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();

        // addition between ciphertext and messages_2
        ciphertext
            .add_constant_static_encoder_inplace(message_2)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message_1 + message_2, decryption, ciphertext.encoder);
        assert_eq!(precision, ciphertext.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_constant_dynamic_encoder_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, max2) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // encoder
    let encoder = crate::Encoder::new(min1, max1, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message_1: f64 = random_message!(min1, max1);
        let message_2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let mut ciphertext = crate::LWE::encode_encrypt(&secret_key, message_1, &encoder).unwrap();

        // addition between ciphertext and messages_2
        ciphertext
            .add_constant_dynamic_encoder_inplace(message_2)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message_1 + message_2, decryption, ciphertext.encoder);
        assert_eq!(precision, ciphertext.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_opposite_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // a message
        let message: f64 = random_message!(min, max);

        // encode and encrypt
        let mut ciphertext = crate::LWE::encode_encrypt(&secret_key, message, &encoder).unwrap();

        // compute the opposite of the second ciphertext
        ciphertext.opposite_inplace().unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(-message, decryption, ciphertext.encoder);
    }
}

#[test]
fn test_encode_encrypt_x_add_centered_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (min2, _) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // encoders
    let encoder1 = crate::Encoder::new(min, max, precision, padding).unwrap();
    let max2 = min2 + max - min;
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..200 {
        // two messages
        let delta = (max - min) / 4. + encoder1.get_granularity() / 2.;
        let message1: f64 = random_message!(min + delta, min + encoder1.delta - delta);
        let message2: f64 = random_message!(min2 + delta, min2 + encoder2.delta - delta);

        // encode and encrypt
        let mut ciphertext1 = crate::LWE::encode_encrypt(&secret_key, message1, &encoder1).unwrap();
        let ciphertext2 = crate::LWE::encode_encrypt(&secret_key, message2, &encoder2).unwrap();

        // addition between ciphertext1 and ciphertext2
        ciphertext1.add_centered_inplace(&ciphertext2).unwrap();

        // decryption
        let decryption: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(message1 + message2, decryption, ciphertext1.encoder);
        assert_eq!(precision, ciphertext1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_with_padding_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(8, 3);
    padding += 1; // at least one bit

    // encoders
    let encoder1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message1: f64 = random_message!(min1, max1);
        let message2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let mut ciphertext1 = crate::LWE::encode_encrypt(&secret_key, message1, &encoder1).unwrap();
        let ciphertext2 = crate::LWE::encode_encrypt(&secret_key, message2, &encoder2).unwrap();

        // addition between ciphertext and message_2
        ciphertext1.add_with_padding_inplace(&ciphertext2).unwrap();

        // decryption
        let decryption: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(message1 + message2, decryption, ciphertext1.encoder);
        assert_eq!(precision, ciphertext1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_sub_with_padding_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(8, 3);
    padding += 1; // at least one bit

    // encoders
    let encoder1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message1: f64 = random_message!(min1, max1);
        let message2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let mut ciphertext1 = crate::LWE::encode_encrypt(&secret_key, message1, &encoder1).unwrap();
        let ciphertext2 = crate::LWE::encode_encrypt(&secret_key, message2, &encoder2).unwrap();

        // subtraction between ciphertext and messages_2
        ciphertext1.sub_with_padding_inplace(&ciphertext2).unwrap();

        // decryption
        let decryption: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(message1 - message2, decryption, ciphertext1.encoder);
        assert_eq!(precision, ciphertext1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_sub_with_padding_exact_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(8, 3);
    padding += 1; // at least one bit

    // encoders
    let encoder1 = crate::Encoder::new_rounding_context(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new_rounding_context(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message1: f64 = random_message!(min1, max1);
        let message2: f64 = random_message!(min2, max2);
        // encode and encrypt
        let mut ciphertext1 = crate::LWE::encode_encrypt(&secret_key, message1, &encoder1).unwrap();
        let ciphertext2 = crate::LWE::encode_encrypt(&secret_key, message2, &encoder2).unwrap();

        // decrypt messages
        let round_message1: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();
        let round_message2: f64 = ciphertext2.decrypt_decode_round(&secret_key).unwrap();

        // subtraction between ciphertext and messages_2
        ciphertext1
            .sub_with_padding_exact_inplace(&ciphertext2)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(
            round_message1 - round_message2,
            decryption,
            ciphertext1.encoder
        );
        assert_eq!(precision + 1, ciphertext1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_add_with_padding_exact_inplace_x_decrypt() {
    // random settings
    let (min1, max1) = generate_random_interval!();
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1; // same interval size
    let (precision, mut padding) = generate_precision_padding!(5, 2);
    padding += 1; // at least one bit

    // encoders
    let encoder1 = crate::Encoder::new_rounding_context(min1, max1, precision, padding).unwrap();
    let encoder2 = crate::Encoder::new_rounding_context(min2, max2, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message1: f64 = random_message!(min1, max1);
        let message2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let mut ciphertext1 = crate::LWE::encode_encrypt(&secret_key, message1, &encoder1).unwrap();
        let ciphertext2 = crate::LWE::encode_encrypt(&secret_key, message2, &encoder2).unwrap();

        // decrypt messages
        let round_message1: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();
        let round_message2: f64 = ciphertext2.decrypt_decode_round(&secret_key).unwrap();

        // subtraction between ciphertext and messages_2
        ciphertext1
            .add_with_padding_exact_inplace(&ciphertext2)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext1.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(
            round_message1 + round_message2,
            decryption,
            ciphertext1.encoder
        );
        assert_eq!(precision + 1, ciphertext1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_mul_constant_static_encoder_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_centered_interval!();
    let (precision, padding) = generate_precision_padding!(6, 2);
    let b = min.abs().min(max.abs()) / 20.;

    // encoders
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // two messages
        let message1: f64 = random_message!(-b, b);
        let message2_float: f64 = random_message!(-b, b);
        let message2: i32 = message2_float as i32;

        // encode and encrypt
        let mut ciphertext = crate::LWE::encode_encrypt(&secret_key, message1, &encoder).unwrap();

        // multiplication between ciphertext and messages2
        ciphertext
            .mul_constant_static_encoder_inplace(message2)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(message1 * (message2 as f64), decryption, ciphertext.encoder);
        assert_eq!(precision, ciphertext.encoder.nb_bit_precision);
    }
}

#[test]
fn test_encode_encrypt_x_mul_constant_with_padding_inplace_x_decrypt() {
    // random settings
    let (min, max) = generate_random_centered_interval!();
    let precision: usize = random_index!(5) + 3;
    let padding = random_index!(3) + precision;
    let nb_bit_padding_mult = precision;
    let b = (random_index!(300) + 3) as f64;

    // encoders
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_2048);

    for _ in 0..100 {
        // two of message
        let message1: f64 = random_message!(min, max);
        let message2: f64 = random_message!(-b, b);

        // encode and encrypt
        let mut ciphertext = crate::LWE::encode_encrypt(&secret_key, message1, &encoder).unwrap();

        // multiplication between ciphertext and messages2
        ciphertext
            .mul_constant_with_padding_inplace(message2, b, nb_bit_padding_mult)
            .unwrap();

        // decryption
        let decryption: f64 = ciphertext.decrypt_decode_round(&secret_key).unwrap();

        // check the precision loss related to the encryption
        assert_eq_granularity!(message1 * message2, decryption, ciphertext.encoder);
    }
}

#[test]
#[ignore]
fn test_encode_encrypt_x_keyswitch_x_decrypt() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(4, 1);
    let base_log = 9;
    let level = 7;

    // encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generate two secret keys
    let secret_key_before = crate::LWESecretKey::new(&crate::LWE128_1024);
    let secret_key_after = crate::LWESecretKey::new(&crate::LWE128_800);

    // generate the key switching key
    let ksk = crate::LWEKSK::new(&secret_key_before, &secret_key_after, base_log, level);

    for _ in 0..20 {
        // a random message
        let message: f64 = random_message!(min, max);
        let ciphertext_before =
            crate::LWE::encode_encrypt(&secret_key_before, message, &encoder).unwrap();

        // key switch
        let ciphertext_after = ciphertext_before.keyswitch(&ksk).unwrap();

        // decryption
        let decryption: f64 = ciphertext_after
            .decrypt_decode_round(&secret_key_after)
            .unwrap();

        // test
        assert_eq_granularity!(message, decryption, ciphertext_after.encoder);
    }
}

#[test]
fn test_encode_encrypt_x_bootstrap_x_decrypt() {
    // random settings
    let (min, max) = (0., 7.);
    let padding: usize = 1;
    let precision: usize = 3;
    let base_log: usize = 7;
    let level: usize = 3;

    // encoders
    let encoder_input = crate::Encoder::new(min, max, precision, padding).unwrap();

    // secret keys
    let rlwe_secret_key = crate::RLWESecretKey::new(&crate::RLWE128_1024_1);
    let secret_key_input = crate::LWESecretKey::new(&crate::LWE128_630);
    let secret_key_output = rlwe_secret_key.to_lwe_secret_key();

    // bootstrapping key
    let bootstrapping_key =
        crate::LWEBSK::new(&secret_key_input, &rlwe_secret_key, base_log, level);

    for _ in 0..200 {
        // a random message
        let message: f64 = random_message!(min, max);

        // encode and encrypt
        let ciphertext_input =
            crate::LWE::encode_encrypt(&secret_key_input, message, &encoder_input).unwrap();

        // bootstrap
        let ciphertext_output = ciphertext_input.bootstrap(&bootstrapping_key).unwrap();

        // decrypt
        let decryption2 = ciphertext_output
            .decrypt_decode(&secret_key_output)
            .unwrap();
        assert_eq_granularity!(message, decryption2, ciphertext_output.encoder);
    }
}

#[test]
fn test_encode_encrypt_x_mul_from_bootstrap_x_decrypt() {
    // random settings for the first encoder and some messages
    let (min1, max1) = generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, 5, 2).unwrap();

    // random settings for the second encoder and some messages
    let (min2, _max2) = generate_random_interval!();
    let max2 = min2 + max1 - min1;
    let encoder_2 = crate::Encoder::new(min2, max2, 5, 2).unwrap();

    // generate a secret key
    let rlwe_secret_key = crate::RLWESecretKey::new(&crate::RLWE128_1024_1);
    let secret_key_input = crate::LWESecretKey::new(&crate::LWE128_630);
    let secret_key_output = rlwe_secret_key.to_lwe_secret_key();

    // bootstrapping key
    let bsk = crate::LWEBSK::new(&secret_key_input, &rlwe_secret_key, 5, 3);

    for _ in 0..10 {
        // random messages
        let message_1: f64 = random_message!(min1, max1);
        let message_2: f64 = random_message!(min2, max2);

        // encode and encrypt
        let ciphertext_1 =
            crate::LWE::encode_encrypt(&secret_key_input, message_1, &encoder_1).unwrap();
        let ciphertext_2 =
            crate::LWE::encode_encrypt(&secret_key_input, message_2, &encoder_2).unwrap();

        // multiplication
        let ciphertext_res = ciphertext_1
            .mul_from_bootstrap(&ciphertext_2, &bsk)
            .unwrap();

        // decrypt
        let decryption = ciphertext_res
            .decrypt_decode_round(&secret_key_output)
            .unwrap();

        // test
        assert_eq_granularity!(message_1 * message_2, decryption, ciphertext_res.encoder);
    }
}

#[test]
fn test_encode_encrypt_x_add_with_new_min_inplace_x_decrypt() {
    // random number of messages
    let (precision, padding) = generate_precision_padding!(8, 8);

    // random settings for the first encoder and some messages
    let (min1, max1) = generate_random_interval!();
    let encoder_1 = crate::Encoder::new(min1, max1, precision, padding).unwrap();

    // random settings for the second encoder and some random messages
    let (min2, _max2) = generate_random_interval!();
    let encoder_2 =
        crate::Encoder::new(min2, min2 + encoder_1.get_size(), precision, padding).unwrap();

    // generate a secret key
    let secret_key = crate::LWESecretKey::new(&crate::LWE128_1024);

    for _ in 0..100 {
        // random messages
        let message_1: f64 = random_message!(min1 + encoder_1.get_size() / 2., max1);
        let message_2: f64 = random_message!(min2, min2 + encoder_1.get_size() / 2.);

        // encode and encrypt
        let mut ciphertext_1 =
            crate::LWE::encode_encrypt(&secret_key, message_1, &encoder_1).unwrap();
        let ciphertext_2 = crate::LWE::encode_encrypt(&secret_key, message_2, &encoder_2).unwrap();

        // new_min
        let new_min: f64 = min1 + min2 + encoder_1.get_size() / 2.;

        // addition between ciphertext_1 and ciphertext_2
        ciphertext_1
            .add_with_new_min_inplace(&ciphertext_2, new_min)
            .unwrap();

        // decryption
        let decryption = ciphertext_1.decrypt_decode_round(&secret_key).unwrap();

        // test
        assert_eq_granularity!(message_1 + message_2, decryption, ciphertext_1.encoder);
        assert_eq!(precision, ciphertext_1.encoder.nb_bit_precision);
    }
}

#[test]
fn test_valid_lookup_table() {
    // generate a random encoder and encode the min value, encrypt it
    // use a constant function in a bootstrapp
    // check the error is the decrypted
    // also do the same thing with a random message

    // random settings
    let (min, max) = generate_random_interval!();
    let padding: usize = random_index!(3) + 1;
    let precision: usize = random_index!(3) + 1;
    let base_log: usize = random_index!(3) + 7;
    let level: usize = random_index!(1) + 3;

    // encoders
    let encoder_input = crate::Encoder::new(min, max, precision, padding).unwrap();
    let encoder_output = crate::Encoder::new(0., 7., 3, 0).unwrap();

    // secret keys
    let rlwe_secret_key = crate::RLWESecretKey::new(&crate::RLWE128_1024_1);
    let secret_key_input = crate::LWESecretKey::new(&crate::LWE128_630);
    let secret_key_output = rlwe_secret_key.to_lwe_secret_key();

    // bootstrapping key
    let bootstrapping_key =
        crate::LWEBSK::new(&secret_key_input, &rlwe_secret_key, base_log, level);

    for _ in 0..100 {
        // a random message
        let zero: f64 = min;
        let message: f64 = random_message!(min, max);

        // a random constant
        let cst: f64 = random_message!(0., 7.).round();

        // encode and encrypt
        let ciphertext_zero =
            crate::LWE::encode_encrypt(&secret_key_input, zero, &encoder_input).unwrap();
        let ciphertext_input =
            crate::LWE::encode_encrypt(&secret_key_input, message, &encoder_input).unwrap();

        // bootstrap
        let ciphertext_output_zero = ciphertext_zero
            .bootstrap_with_function(&bootstrapping_key, |_| cst, &encoder_output)
            .unwrap();
        let ciphertext_output = ciphertext_input
            .bootstrap_with_function(&bootstrapping_key, |_| cst, &encoder_output)
            .unwrap();

        // decrypt
        let decryption2 = ciphertext_output
            .decrypt_decode_round(&secret_key_output)
            .unwrap();
        assert_eq_granularity!(cst, decryption2, ciphertext_output.encoder);

        // decrypt
        let decryption_zero = ciphertext_output_zero
            .decrypt_decode_round(&secret_key_output)
            .unwrap();
        assert_eq_granularity!(cst, decryption_zero, ciphertext_output_zero.encoder);
    }
}
