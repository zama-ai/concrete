use itertools::izip;

#[test]
fn test_decode_nth() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let m: f64 = random_message!(min, max);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();
    let plaintext = encoder.encode_single(m).unwrap();
    let decoding = plaintext.decode_nth(0).unwrap();

    // test
    assert_eq_granularity!(m, decoding, encoder);
}

#[test]
fn test_new_encode_x_decode() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let messages: Vec<f64> = random_messages!(min, max, 10);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // encode and decode
    let plaintext = crate::Plaintext::encode(&messages, &encoder).unwrap();
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
        assert_eq_granularity!(m, d, e);
    }
}

// #[test]
// fn test_new_single_encoder_x_encode_x_decode() {
//     // random settings
//     let (min, max) = generate_random_interval!();
//     let (precision, padding) = generate_precision_padding!(8, 8);

//     // generates a random message
//     let messages: Vec<f64> = random_messages!(min, max, 10);

//     // create an encoder
//     let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

//     // encode and decode
//     let mut plaintext = crate::Plaintext::new_single_encoder(messages.len(), &encoder);
//     plaintext.encode(&messages).unwrap();
//     let decoding = plaintext.decode().unwrap();

//     // test
//     for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
//         assert_eq_granularity!(m, d, e);
//     }
// }

#[test]
fn test_zero_x_set_encoders_x_encode_x_decode() {
    let nb_messages: usize = 10;
    let mut encoders: Vec<crate::Encoder> = vec![crate::Encoder::zero(); nb_messages];
    let mut messages: Vec<f64> = vec![0.; nb_messages];

    for (m, e) in izip!(messages.iter_mut(), encoders.iter_mut()) {
        // random settings
        let (min, max) = generate_random_interval!();
        let (precision, padding) = generate_precision_padding!(8, 8);

        // generates a random message
        *m = random_message!(min, max);
        println!("[{},{}] -> m {}", min, max, m);

        // create an encoder
        e.copy(&crate::Encoder::new(min, max, precision, padding).unwrap());
        println!("encoder {}", e);
    }

    // encode and decode
    let mut plaintext = crate::Plaintext::zero(nb_messages);
    plaintext.set_encoders(&encoders);
    plaintext.encode_inplace(&messages).unwrap();
    let decoding = plaintext.decode().unwrap();

    // test
    println!("plaintext: {}", plaintext);
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
        assert_eq_granularity!(m, d, e);
    }
}

#[test]
fn test_zero_x_set_encoders_from_one_x_encode_x_decode() {
    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // generates a random message
    let messages: Vec<f64> = random_messages!(min, max, 10);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();

    // encode and decode
    let mut plaintext = crate::Plaintext::zero(messages.len());
    plaintext.set_encoders_from_one(&encoder);
    plaintext.encode_inplace(&messages).unwrap();
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
        assert_eq_granularity!(m, d, e);
    }
}

#[test]
fn test_zero_x_set_nth_encoder_from_one_x_encode_x_decode() {
    let nb_messages: usize = 10;

    // random settings
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);

    // random index
    let index: usize = random_index!(nb_messages);

    // generates a random message
    let mut messages: Vec<f64> = random_messages!(min, max, nb_messages);

    // create an encoder
    let encoder = crate::Encoder::new(min, max, precision, padding).unwrap();
    let mut plaintext = crate::Plaintext::zero(messages.len());
    plaintext.set_encoders_from_one(&encoder);

    // another random encoder and message
    let (min, max) = generate_random_interval!();
    let (precision, padding) = generate_precision_padding!(8, 8);
    let encoder_bis = crate::Encoder::new(min, max, precision, padding).unwrap();
    plaintext.set_nth_encoder(index, &encoder_bis);
    messages[index] = random_message!(min, max);

    // encode and decode
    plaintext.encode_inplace(&messages).unwrap();
    let decoding = plaintext.decode().unwrap();

    // test
    for (m, d, e) in izip!(messages.iter(), decoding.iter(), plaintext.encoders.iter()) {
        assert_eq_granularity!(m, d, e);
    }
}
