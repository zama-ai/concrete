macro_rules! lwe_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {

            use crate::core_api::crypto::SecretKey;
            use crate::core_api::crypto::{lwe, LWE};
            use crate::core_api::math::{Random, Tensor};
            use crate::npe;
            use crate::types::N_TESTS;
            use crate::Types;
            use rand::Rng;

            type Torus = $T;

            #[test]
            fn test_key_switch() {
                //! create a KSK and key switch some LWE samples
                //! warning: not a randomized test for the parameters

                // fix a set of parameters
                let n_bit_msg = 8; // bit precision of the plaintext
                let nb_ct: usize = N_TESTS; // number of messages to encrypt
                let base_log: usize = 3; // a parameter of the gadget matrix
                let level: usize = 8; // a parameter of the gadget matrix
                let mut messages: Vec<Torus> = vec![0; nb_ct]; // the set of messages to encrypt
                let std_input = f64::powi(2., -10); // standard deviation of the encrypted messages to KS
                let std_ksk = f64::powi(2., -25); // standard deviation of the ksk

                // fill the messages with random Torus element set to zeros but the n_bit_msg MSB uniformly picked
                Random::rng_uniform_n_msb(&mut messages, n_bit_msg);

                // set parameters related to the before (stands for 'before the KS')
                let dimension_before = 1024;
                let sk_len_before: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension_before, 1);

                // set parameters related to the after (stands for 'after the KS')
                let dimension_after = 600;
                let sk_len_after: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension_after, 1);

                // create and fill the before and the after keys with random bits
                let mut sk_before: Vec<Torus> = vec![0; sk_len_before];
                Tensor::uniform_random_default(&mut sk_before);
                let mut sk_after: Vec<Torus> = vec![0; sk_len_after];
                Tensor::uniform_random_default(&mut sk_after);

                // create the before ciphertexts and the after ciphertexts
                let mut ciphertexts_before: Vec<Torus> = vec![0; nb_ct * (dimension_before + 1)];
                let mut ciphertexts_after: Vec<Torus> = vec![0; nb_ct * (dimension_after + 1)];

                // key switching key generation
                let mut ksk: Vec<Torus> =
                    vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];

                LWE::create_key_switching_key(
                    &mut ksk,
                    base_log,
                    level,
                    dimension_after,
                    std_ksk,
                    &sk_before,
                    &sk_after,
                );

                // encrypts with the before key our messages
                LWE::sk_encrypt(
                    &mut ciphertexts_before,
                    &sk_before,
                    &messages,
                    dimension_before,
                    std_input,
                );

                // key switch before -> after
                LWE::key_switch(
                    &mut ciphertexts_after,
                    &ciphertexts_before,
                    &ksk,
                    base_log,
                    level,
                    dimension_before,
                    dimension_after,
                );

                // decryption with the after key
                let mut dec_messages: Vec<Torus> = vec![0; nb_ct];
                LWE::compute_phase(
                    &mut dec_messages,
                    &sk_after,
                    &ciphertexts_after,
                    dimension_after,
                );

                // calls the NPE to find out the amount of noise after KS
                let output_variance = <$T as npe::LWE>::key_switch(
                    dimension_before,
                    level,
                    base_log,
                    std_ksk * std_ksk,
                    std_input * std_input,
                );

                if nb_ct < 7 {
                    // assert the difference between the original messages and the decrypted messages
                    assert_delta_std_dev!(messages, dec_messages, f64::sqrt(output_variance));
                } else {
                    assert_noise_distribution!(
                        messages,
                        dec_messages,
                        output_variance,
                        "keyswitch"
                    );
                }
            }

            #[test]
            fn test_mono_key_switch() {
                //! create a KSK and key switch one LWE sample
                //! warning: not a randomized test for the parameters

                // fix a set of parameters
                let n_bit_msg = 8; // bit precision of the plaintext
                let base_log: usize = 3; // a parameter of the gadget matrix
                let level: usize = 8; // a parameter of the gadget matrix
                let mut messages: Vec<Torus> = vec![0; 1]; // the message to encrypt
                let std_input = f64::powi(2., -10); // standard deviation of the encrypted messages to KS
                let std_ksk = f64::powi(2., -25); // standard deviation of the ksk

                // fill the messages with random Torus element set to zeros but the n_bit_msg MSB uniformly picked
                Random::rng_uniform_n_msb(&mut messages, n_bit_msg);

                // set parameters related to the before (stands for 'before the KS')
                let dimension_before = 1024;
                let sk_len_before: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension_before, 1);

                // set parameters related to the after (stands for 'after the KS')
                let dimension_after = 600;
                let sk_len_after: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension_after, 1);

                // create and fill the before and the after keys with random bits
                let mut sk_before: Vec<Torus> = vec![0; sk_len_before];
                Tensor::uniform_random_default(&mut sk_before);
                let mut sk_after: Vec<Torus> = vec![0; sk_len_after];
                Tensor::uniform_random_default(&mut sk_after);

                // create the before ciphertexts and the after ciphertexts
                let mut ciphertexts_before: Vec<Torus> = vec![0; dimension_before + 1];
                let mut ciphertexts_after: Vec<Torus> = vec![0; dimension_after + 1];

                // key switching key generation
                let mut ksk: Vec<Torus> =
                    vec![0; lwe::get_ksk_size(dimension_before, dimension_after, level)];

                LWE::create_key_switching_key(
                    &mut ksk,
                    base_log,
                    level,
                    dimension_after,
                    std_ksk,
                    &sk_before,
                    &sk_after,
                );

                // encrypts with the before key our messages
                LWE::sk_encrypt(
                    &mut ciphertexts_before,
                    &sk_before,
                    &messages,
                    dimension_before,
                    std_input,
                );

                // key switch before -> after
                LWE::mono_key_switch(
                    &mut ciphertexts_after,
                    &ciphertexts_before,
                    &ksk,
                    base_log,
                    level,
                    dimension_before,
                    dimension_after,
                );

                // decryption with the after key
                let mut dec_messages: Vec<Torus> = vec![0; 1];
                LWE::compute_phase(
                    &mut dec_messages,
                    &sk_after,
                    &ciphertexts_after,
                    dimension_after,
                );

                // calls the NPE to find out the amount of noise after KS
                let max_variance = <$T as npe::LWE>::key_switch(
                    dimension_before,
                    level,
                    base_log,
                    std_ksk * std_ksk,
                    std_input * std_input,
                );

                // assert the difference between the original messages and the decrypted messages
                assert_delta_std_dev!(messages, dec_messages, f64::sqrt(max_variance));
            }

            #[test]
            fn test_compute_phase_randomized() {
                //! encrypts a bunch of messages and decrypts them
                //! warning: std_dev is not randomized
                //! only assert with assert_delta_std_dev
                let mut rng = rand::thread_rng();

                // generate random settings
                let mut nb_ct: usize = rng.gen();
                nb_ct = (nb_ct % 100) + 1;
                let mut dimension: usize = rng.gen();
                dimension = (dimension % 1000) + 1;
                let std_dev: f64 = f64::powi(2., -25);

                // generate the secret key
                let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Tensor::uniform_random_default(&mut sk);

                // generate random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct];
                Tensor::uniform_random_default(&mut messages);

                // creation of tensors for our ciphertexts
                let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];

                // encryption
                LWE::sk_encrypt(&mut ciphertexts, &sk, &messages, dimension, std_dev);

                // creation of a tensor for our decrypted messages
                let mut decryptions: Vec<Torus> = vec![0; nb_ct];

                // decryption
                LWE::compute_phase(&mut decryptions, &sk, &ciphertexts, dimension);

                // make sure that after decryption we recover the original plaintext
                assert_delta_std_dev!(messages, decryptions, std_dev);
            }

            #[test]
            fn test_multisum_npe() {
                //! encrypts messages, does a multisum and decrypts the result
                //! warning: std_dev is not randomized
                let mut rng = rand::thread_rng();
                let mut new_msg: Vec<Torus> = vec![0; N_TESTS];
                let mut msg: Vec<Torus> = vec![0; N_TESTS];

                // generate random settings
                let mut nb_ct: usize = rng.gen();
                nb_ct = (nb_ct % 100) + 1;
                let mut dimension = rng.gen();
                dimension = (dimension % 1000) + 1;
                let std: f64 = f64::powi(2., -25);

                // generate random weights
                let mut weights: Vec<Torus> = vec![0; nb_ct];
                let mut s_weights: Vec<<Torus as Types>::STorus> = vec![0; nb_ct];
                let mut rng = rand::thread_rng();
                for (w, sw) in weights.iter_mut().zip(s_weights.iter_mut()) {
                    *sw = rng.gen::<Torus>().rem_euclid(512) as <Torus as Types>::STorus - 256;
                    *w = *sw as Torus;
                }
                let bias = rng.gen::<Torus>().rem_euclid(1024);

                for i in 0..N_TESTS {
                    // generate the secret key
                    let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
                    let mut sk: Vec<Torus> = vec![0; sk_len];
                    Tensor::uniform_random_default(&mut sk);

                    // generate random messages
                    let mut messages: Vec<Torus> = vec![0; nb_ct];
                    Tensor::uniform_random_default(&mut messages);

                    // generate trivial encryptions for the witness
                    let mut witness: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                    // for i in 0..nb_ct {
                    //     witness[i * (dimension )] = messages[i] ;
                    // }
                    LWE::trivial_sk_encrypt(&mut witness, &messages, dimension, std);
                    // generate ciphertexts with the secret key
                    let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                    LWE::sk_encrypt(&mut ciphertexts, &sk, &messages, dimension, std);

                    // allocation for the results
                    let mut ct_res: Vec<Torus> = vec![0; dimension + 1];
                    let mut ct_res_witness: Vec<Torus> = vec![0; dimension + 1];

                    // computation of the multisums
                    LWE::mono_multisum_with_bias(
                        &mut ct_res,
                        &ciphertexts,
                        &weights,
                        bias,
                        dimension,
                    );

                    LWE::mono_multisum_with_bias(
                        &mut ct_res_witness,
                        &witness,
                        &weights,
                        bias,
                        dimension,
                    );
                    // decryption
                    LWE::compute_phase(&mut new_msg[i..i + 1], &sk, &ct_res, dimension);
                    msg[i] = ct_res_witness[dimension];
                }

                // noise prediction
                let mut weights: Vec<Torus> = vec![0; s_weights.len()];
                for (w, sw) in weights.iter_mut().zip(s_weights.iter()) {
                    *w = *sw as Torus;
                }
                let output_variance: f64 = <Torus as npe::LWE>::multisum_uncorrelated(
                    &vec![f64::powi(std, 2); nb_ct],
                    &weights,
                );

                if N_TESTS < 7 {
                    assert_delta_std_dev!(new_msg, msg, f64::sqrt(output_variance));
                } else {
                    assert_noise_distribution!(msg, new_msg, output_variance, "multisum");
                }
            }

            #[test]
            fn test_encrypt_decrypt() {
                //! encrypts a bunch of messages and decrypts them
                //! warning: std_dev is not randomized
                // settings
                let nb_ct: usize = 100000; // N_TESTS;
                let dimension = 600;
                let std_dev = f64::powi(2., -15);

                // generate the secret key
                let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Tensor::uniform_random_default(&mut sk);

                // generate random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct];
                Tensor::uniform_random_default(&mut messages);

                // encryption
                let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                LWE::sk_encrypt(&mut ciphertexts, &sk, &messages, dimension, std_dev);

                // decryption
                let mut decryptions: Vec<Torus> = vec![0; nb_ct];
                LWE::compute_phase(&mut decryptions, &sk, &ciphertexts, dimension);

                // tests
                if nb_ct < 7 {
                    // assert the difference between the original messages and the decrypted messages
                    assert_delta_std_dev!(messages, decryptions, std_dev);
                } else {
                    assert_noise_distribution!(
                        messages,
                        decryptions,
                        f64::powi(std_dev, 2),
                        "encryption"
                    );
                }
            }

            #[test]
            fn test_scalar_mul() {
                //! encrypts a bunch of messages, performs a scalar multiplication and decrypts them
                //! the settings are not randomized
                let mut rng = rand::thread_rng();

                // settings
                let nb_ct: usize = N_TESTS;
                let dimension = 600;
                let std_dev = f64::powi(2., -15);

                // generate the secret key
                let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Tensor::uniform_random_default(&mut sk);

                // generate random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct];
                Tensor::uniform_random_default(&mut messages);

                // encryption
                let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                LWE::sk_encrypt(&mut ciphertexts, &sk, &messages, dimension, std_dev);

                // generate a random signed weight vector represented as Torus elements
                let mut weight: <Torus as Types>::STorus = rng.gen();
                weight %= 512; // between -511 and 511
                let w: Vec<Torus> = vec![weight as Torus; nb_ct];

                // scalar mul
                let mut ciphertexts_sm: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                LWE::scalar_mul(&mut ciphertexts_sm, &ciphertexts, &w, dimension);

                // compute on cleartexts the multiplication
                let mut messages_mul: Vec<Torus> = vec![0; nb_ct];
                for (mm, (w_i, m)) in messages_mul.iter_mut().zip(w.iter().zip(messages.iter())) {
                    *mm = (*w_i as Torus).wrapping_mul(*m);
                }

                // decryption
                let mut decryptions: Vec<Torus> = vec![0; nb_ct];
                LWE::compute_phase(&mut decryptions, &sk, &ciphertexts_sm, dimension);

                // test
                let output_variance: f64 =
                    <Torus as npe::LWE>::single_scalar_mul(f64::powi(std_dev, 2), weight as Torus);
                if nb_ct < 7 {
                    // assert the difference between the original messages and the decrypted messages
                    assert_delta_std_dev!(messages_mul, decryptions, f64::sqrt(output_variance));
                } else {
                    assert_noise_distribution!(
                        messages_mul,
                        decryptions,
                        output_variance,
                        "scalarmul"
                    );
                }
            }

            #[test]
            fn test_scalar_mul_random() {
                //! encrypts a bunch of messages, performs a scalar multiplication and decrypts them
                //! warning: std_dev is not randomized
                //! only assert with assert_delta_std_dev
                let mut rng = rand::thread_rng();

                // settings
                let mut nb_ct: usize = rng.gen();
                nb_ct = (nb_ct % 100) + 1;
                let mut dimension = rng.gen();
                dimension = (dimension % 1000) + 1;
                let std_dev = f64::powi(2., -15);

                // generate the secret key
                let sk_len: usize = <Torus as SecretKey>::get_secret_key_length(dimension, 1);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Tensor::uniform_random_default(&mut sk);

                // generate random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct];
                Tensor::uniform_random_default(&mut messages);

                // encryption
                let mut ciphertexts: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                LWE::sk_encrypt(&mut ciphertexts, &sk, &messages, dimension, std_dev);

                // generate a random signed weight vector as Torus elements
                let mut w: Vec<Torus> = vec![0; nb_ct];
                for w_i in w.iter_mut() {
                    let mut tmp: <Torus as Types>::STorus = rng.gen();
                    tmp %= 512;
                    *w_i = tmp as Torus;
                }

                // scalar mul
                let mut ciphertexts_sm: Vec<Torus> = vec![0; nb_ct * (dimension + 1)];
                LWE::scalar_mul(&mut ciphertexts_sm, &ciphertexts, &w, dimension);

                // compute on cleartexts the multiplication
                let mut messages_mul: Vec<Torus> = vec![0; nb_ct];
                for (mm, (w_i, m)) in messages_mul.iter_mut().zip(w.iter().zip(messages.iter())) {
                    *mm = (*w_i as Torus).wrapping_mul(*m);
                }

                // decryption
                let mut decryptions: Vec<Torus> = vec![0; nb_ct];
                LWE::compute_phase(&mut decryptions, &sk, &ciphertexts_sm, dimension);

                // test
                for (mm, (d, w_i)) in messages_mul
                    .chunks(1)
                    .zip(decryptions.chunks(1).zip(w.iter()))
                {
                    // noise prediction work
                    let output_variance: f64 =
                        <Torus as npe::LWE>::single_scalar_mul(f64::powi(std_dev, 2), *w_i);
                    assert_delta_std_dev!(mm, d, f64::sqrt(output_variance));
                }
            }
        }
    };
}

lwe_test_mod!(u32, tests_u32);
lwe_test_mod!(u64, tests_u64);
