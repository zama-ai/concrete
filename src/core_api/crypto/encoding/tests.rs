macro_rules! encoding_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::core_api::crypto::Encoding;
            use crate::core_api::math::Tensor;
            use crate::Types;
            use rand::Rng;
            type Torus = $T;

            #[test]
            fn test_encoding_decoding() {
                //! Encodes and decodes random messages
                let mut rng = rand::thread_rng();
                let n_tests = 1000;
                for _i in 0..n_tests {
                    // the real interval is [int_o,int_beta]
                    let mut int_o: Torus = rng.gen();
                    let mut int_beta: Torus = rng.gen();

                    // if int_o > int_beta, we swap them
                    if int_beta < int_o {
                        std::mem::swap(&mut int_beta, &mut int_o);
                    }

                    // converts int_o and int_delta into f64
                    let o: f64 = int_o as f64;
                    let delta: f64 = (int_beta - int_o) as f64;

                    // generates a random message
                    let mut int_m: Torus = rng.gen();
                    int_m = (int_m % (int_beta - int_o)) + int_o;
                    let m: f64 = int_m as f64;

                    // encodes and decodes
                    let encoding: Torus = Encoding::encode(m, o, delta);
                    let decoding: f64 = Encoding::decode(encoding, o, delta);

                    // test
                    if <$T as Types>::TORUS_BIT == 32 {
                        assert_delta_scalar!(m, decoding, 1);
                    } else {
                        assert_delta_scalar!(m, decoding, 1 << 10);
                    }
                }
            }

            #[test]
            fn test_several_encoding_decoding() {
                //! Encodes and decodes random messages
                let n_tests = 1000; // number of encode decode tests

                // we generate n_tests random intervals [int_o,int_beta[
                let mut int_os: Vec<Torus> = vec![0 as Torus; n_tests];
                Tensor::uniform_random_default(&mut int_os);
                let mut int_betas: Vec<Torus> = vec![0 as Torus; n_tests];
                Tensor::uniform_random_default(&mut int_betas);

                let mut os: Vec<f64> = vec![0.; n_tests];

                // generates random messages
                let mut int_messages: Vec<Torus> = vec![0 as Torus; n_tests];
                Tensor::uniform_random_default(&mut int_messages);

                let mut messages: Vec<f64> = vec![0.; n_tests];
                let mut deltas: Vec<f64> = vec![0.; n_tests];
                // if int_o > int_beta, we swap them
                for (int_beta, int_o, o, int_m, m, delta) in izip!(
                    int_betas.iter_mut(),
                    int_os.iter_mut(),
                    os.iter_mut(),
                    int_messages.iter(),
                    messages.iter_mut(),
                    deltas.iter_mut()
                ) {
                    if *int_beta < *int_o {
                        std::mem::swap(&mut (*int_beta), &mut (*int_o));
                    }
                    // converts int_o and int_delta into f64
                    // *beta = *int_beta as f64;
                    *o = *int_o as f64;
                    *m = ((*int_m % (*int_beta - *int_o)) + *int_o) as f64;
                    *delta = (*int_beta - *int_o) as f64;
                }

                // encodes and decodes
                let mut encodings: Vec<Torus> = vec![0 as Torus; n_tests];
                let mut decodings: Vec<f64> = vec![0. as f64; n_tests];

                Encoding::several_encode(&mut encodings, &messages, &os, &deltas);
                Encoding::several_decode(&mut decodings, &encodings, &os, &deltas);

                // test
                if <$T as Types>::TORUS_BIT == 32 {
                    assert_delta!(messages, decodings, 1);
                } else {
                    assert_delta!(messages, decodings, 1 << 11);
                }
            }

            #[test]
            fn test_several_encoding_decoding_with_same_parameters() {
                //! Encodes and decodes random messages
                let n_tests = 1000;

                // the real interval is [int_o,int_beta]
                let mut int_os: Vec<Torus> = vec![0 as Torus; 1];
                Tensor::uniform_random_default(&mut int_os);
                let mut int_o = int_os[0];
                let mut int_betas: Vec<Torus> = vec![0 as Torus; 1];
                let mut int_beta = int_betas[0];

                Tensor::uniform_random_default(&mut int_betas);

                // generates random messages
                let mut int_messages: Vec<Torus> = vec![0 as Torus; n_tests];
                Tensor::uniform_random_default(&mut int_messages);

                let mut messages: Vec<f64> = vec![0.; n_tests];
                let delta: f64;

                // if int_o > int_beta, we swap them
                if int_beta < int_o {
                    std::mem::swap(&mut int_beta, &mut int_o);
                }

                let o: f64 = int_o as f64;
                delta = (int_beta - int_o) as f64;

                for (int_m, m) in izip!(int_messages.iter(), messages.iter_mut(),) {
                    *m = ((*int_m % (int_beta - int_o)) + int_o) as f64;
                }

                // encodes and decodes
                let mut encodings: Vec<Torus> = vec![0 as Torus; n_tests];
                let mut decodings: Vec<f64> = vec![0. as f64; n_tests];

                Encoding::several_encode_with_same_parameters(&mut encodings, &messages, o, delta);
                Encoding::several_decode_with_same_parameters(&mut decodings, &encodings, o, delta);

                // test
                if <$T as Types>::TORUS_BIT == 32 {
                    assert_delta!(messages, decodings, 1);
                } else {
                    assert_delta!(messages, decodings, 1 << 10);
                }
            }
        }
    };
}

encoding_test_mod!(u32, tests_u32);
encoding_test_mod!(u64, tests_u64);
