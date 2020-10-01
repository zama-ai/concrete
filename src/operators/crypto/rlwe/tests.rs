macro_rules! rlwe_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::operators::crypto::SecretKey;
            use crate::operators::crypto::RLWE;
            use crate::operators::math::Random;
            use rand;
            use rand::Rng;

            type Torus = $T;

            #[test]
            fn test_encrypt_decrypt() {
                //! encrypts messages and then decrypts it
                let mut rng = rand::thread_rng();

                // random settings
                let mut nb_ct: usize = rng.gen();
                nb_ct = (nb_ct % 512) + 1;
                let mut dimension: usize = 2;
                dimension = (dimension % 512) + 1;
                let mut polynomial_size: usize = 2;
                polynomial_size = (polynomial_size % 512) + 1;
                let std_dev = f64::powi(2., -20);

                // generates a secret key
                let sk_len: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Random::rng_uniform(&mut sk);

                // generates random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct * polynomial_size];
                Random::rng_uniform(&mut messages);

                // encrypts
                let mut ciphertexts: Vec<Torus> =
                    vec![0; nb_ct * (dimension + 1) * polynomial_size];

                RLWE::sk_encrypt(
                    &mut ciphertexts,
                    &sk,
                    &messages,
                    dimension as usize,
                    polynomial_size,
                    std_dev,
                );

                // decrypts
                let mut decryptions: Vec<Torus> = vec![0; nb_ct * polynomial_size];
                RLWE::compute_phase(
                    &mut decryptions,
                    &sk,
                    &ciphertexts,
                    dimension,
                    polynomial_size,
                );

                // test
                assert_delta_std_dev!(messages, decryptions, std_dev);
            }

            #[test]
            fn test_encrypt_decrypt_2() {
                //! encrypts messages (polynomial by polynomial) and then decrypts them all
                let mut rng = rand::thread_rng();

                // random settings
                let mut nb_ct: usize = rng.gen();
                nb_ct = (nb_ct % 512) + 1;
                let mut dimension: usize = 2;
                dimension = (dimension % 512) + 1;
                let mut polynomial_size: usize = 2;
                polynomial_size = (polynomial_size % 512) + 1;
                let std_dev = f64::powi(2., -20);

                // generates a secret key
                let sk_len: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                let mut sk: Vec<Torus> = vec![0; sk_len];
                Random::rng_uniform(&mut sk);

                // generates random messages
                let mut messages: Vec<Torus> = vec![0; nb_ct * polynomial_size];
                Random::rng_uniform(&mut messages);

                // encrypts
                let mut ciphertexts: Vec<Torus> =
                    vec![0; nb_ct * (dimension + 1) * polynomial_size];
                for (ciphertext, message) in ciphertexts
                    .chunks_mut((dimension + 1) * polynomial_size)
                    .zip(messages.chunks(polynomial_size))
                {
                    RLWE::sk_encrypt(
                        ciphertext,
                        &sk,
                        message,
                        dimension,
                        polynomial_size,
                        std_dev,
                    );
                }

                // decrypts
                let mut decryptions: Vec<Torus> = vec![0; nb_ct * polynomial_size];
                RLWE::compute_phase(
                    &mut decryptions,
                    &sk,
                    &ciphertexts,
                    dimension,
                    polynomial_size,
                );

                // test
                assert_delta_std_dev!(messages, decryptions, std_dev);
            }
        }
    };
}

rlwe_test_mod!(u32, tests_u32);
rlwe_test_mod!(u64, tests_u64);
