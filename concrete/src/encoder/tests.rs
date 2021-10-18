#![allow(clippy::float_cmp)]

use itertools::izip;

use concrete_core::math::tensor::Tensor;

#[test]
fn test_new_x_encode_single_x_decode_single() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let m: f64 = random_message!(min, max);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // encode and decode
    let plaintext = encoder.encode_single(m).unwrap();
    let decoding = encoder.decode_single(plaintext.plaintexts[0]).unwrap();

    // test
    assert_eq_granularity!(m, decoding, encoder);
}

#[test]
fn test_new_centered_x_encode_single_x_decode_single() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let m: f64 = random_message!(min, max);

    // create an encoder
    let encoder =
        crate::Encoder::new_centered(min + (max - min) / 2., (max - min) / 2., precision, padding)
            .unwrap();

    // encode and decode
    let plaintext = encoder.encode_single(m).unwrap();
    let decoding = encoder.decode_single(plaintext.plaintexts[0]).unwrap();

    // test
    assert_eq_granularity!(m, decoding, encoder);
}

#[test]
fn test_new_x_is_valid() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    //test
    assert!(encoder.is_valid());
}

#[test]
fn test_new_centered_x_is_valid() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // create an encoder
    let encoder =
        crate::Encoder::new_centered(min + (max - min) / 2., (max - min) / 2., precision, padding)
            .unwrap();

    //test
    assert!(encoder.is_valid());
}

#[test]
fn test_zero_x_is_valid() {
    // create a zero encoder
    let encoder = crate::Encoder::zero();

    //test
    assert!(!encoder.is_valid());
}

#[test]
fn test_new_x_encode() {
    let nb_messages: usize = 10;

    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let messages: Vec<f64> = random_messages!(min, max, nb_messages);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // encode and decode
    let plaintext = encoder.encode(&messages).unwrap();
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
        assert_eq_granularity!(m, d, e);
    }
}

#[test]
fn test_new_x_encode_single_x_copy_x_decode_single() {
    // create a first encoder
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // generates a random message
    let m: f64 = random_message!(min, max);

    // create a second encoder
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let mut encoder_copy = crate::Encoder::new(min, max, precision, padding).unwrap();

    // copy the encoder
    encoder_copy.copy(&encoder);

    // encode and decode
    let plaintext = encoder.encode_single(m).unwrap();
    let decoding = encoder_copy.decode_single(plaintext.plaintexts[0]).unwrap();

    // tests
    assert_eq_granularity!(m, decoding, encoder);
    assert_eq!(encoder, encoder_copy);
}

#[test]
fn test_new_rounding_context_x_encode_single_x_decode_single() {
    // create an encoder with a granularity = 1
    let (min, _) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let max = min + f64::powi(2., precision as i32) - 1.;
    let encoder = crate::Encoder::new_rounding_context(min, max, precision, padding).unwrap();

    for _ in 0..100 {
        // generates a random message
        let m: f64 = random_index!((f64::powi(2., precision as i32)) as usize) as f64; // [0,2**prec[ + min
        let m1 = m + min;

        // encode and decode
        let plaintext = encoder.encode_single(m1).unwrap();
        let decoding = encoder.decode_single(plaintext.plaintexts[0]).unwrap();

        // message with error in [-0.5,0.5]
        let m2: f64 = m1
            + if m == 0. {
                random_message!(0., 0.5)
            } else {
                random_message!(-0.5, 0.5)
            };

        // encode and decode
        let plaintext2 = encoder.encode_single(m2).unwrap();
        let decoding2 = encoder.decode_single(plaintext2.plaintexts[0]).unwrap();

        // tests
        assert_eq!(m1, decoding);
        assert_eq!(m1, decoding2);
    }
}

#[test]
fn margins_with_integers() {
    let power: usize = random_index!(5) + 2;
    let nb_messages: usize = (1 << power) - 1;
    let min = 0.;
    let max = f64::powi(2., power as i32) - 1.;
    let padding = random_index!(8);

    // generates a random message
    let mut messages: Vec<f64> = vec![0.; nb_messages];
    for (i, m) in messages.iter_mut().enumerate() {
        *m = i as f64;
    }

    // create an encoder
    let encoder = crate::Encoder::new(min, max, power, padding).unwrap();
    let encoder_round = crate::Encoder::new_rounding_context(min, max, power, padding).unwrap();

    // encode
    let mut plaintext = encoder_round.encode(&messages).unwrap();

    // add some error
    let random_errors = random_messages!(0., 0.5, nb_messages);
    let plaintext_error = encoder.encode(&random_errors).unwrap();
    Tensor::from_container(plaintext.plaintexts.as_mut_slice()).update_with_add(
        &Tensor::from_container(plaintext_error.plaintexts.as_slice()),
    );

    // decode
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), random_errors.iter()) {
        println!("m {} d {} e {} ", m, d, e);
        assert_eq!(m, d);
    }

    // encode
    let mut plaintext = encoder_round.encode(&messages).unwrap();

    // sub some error
    let random_errors = random_messages!(0., 0.5, nb_messages);
    let plaintext_error = encoder.encode(&random_errors).unwrap();
    Tensor::from_container(plaintext.plaintexts.as_mut_slice()).update_with_wrapping_sub(
        &Tensor::from_container(plaintext_error.plaintexts.as_slice()),
    );

    // decode
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), random_errors.iter()) {
        println!("m {} d {} e {} ", m, d, e);
        assert_eq!(m, d);
    }
}

#[test]
fn margins_with_reals() {
    let nb_messages: usize = 400;
    let (min, max) = generate_random_interval!();
    let padding = random_index!(3);
    let precision = random_index!(3) + 2;

    // generates a random message

    let mut messages: Vec<f64> = random_messages!(min, max, nb_messages);
    messages[0] = min;
    messages[1] = max;

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // encode
    let mut plaintext = encoder.encode(&messages).unwrap();

    // add some error
    let random_errors: Tensor<Vec<u64>> = concrete_core::math::random::RandomGenerator::new(None)
        .random_gaussian_tensor(nb_messages, 0., f64::powi(2., -25));
    Tensor::from_container(plaintext.plaintexts.as_mut_slice())
        .update_with_wrapping_add(&random_errors);

    // decode
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d) in izip!(messages.iter(), decoding.iter()) {
        assert!(
            f64::abs(m - d) <= encoder.get_granularity(),
            "error: m {} d {} ",
            m,
            d
        );
    }
}
