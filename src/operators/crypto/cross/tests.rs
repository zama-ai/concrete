macro_rules! cross_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::npe;
            use crate::operators::crypto::{cross, Cross, SecretKey, LWE, RGSW, RLWE};
            use crate::operators::math::Tensor;
            use crate::types::N_TESTS;
            use crate::types::{C2CPlanTorus, CTorus};
            use crate::Types;
            use fftw::array::AlignedVec;
            use fftw::plan::*;
            use fftw::types::*;
            use num_traits::Zero;

            type Torus = $T;
            #[test]
            fn test_bootstrapp_noise() {
                //! test that the bootstrapping noise matches the theoretical noise
                //! This test is design to remove the impact of the drift, we only
                //! check the noise added by the external products

                // fix different polynomial degrees
                let degrees = vec![1024]; // vec![512, 1024, 2048];
                for polynomial_size in degrees {
                    // fix a set of parameters
                    let nb_test: usize = N_TESTS;
                    let dimension: usize = 1;
                    let lwe_dimension: usize = 630;
                    let level = 3;
                    let base_log = 7;
                    let std = f64::powi(2., -29);
                    let lwe_sk_n_bits = lwe_dimension; // as many bit as the length of th e lwe

                    // compute the length of the lwe and rlwe secret keys
                    let lwe_sk_len: usize =
                        <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
                    let rlwe_sk_len: usize =
                        <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);

                    // allocate secret keys
                    let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                    let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];

                    // allocate message vectors
                    let mut msg: Vec<Torus> = vec![0; nb_test];
                    let mut new_msg: Vec<Torus> = vec![0; nb_test];
                    // launch nb_test tests
                    for i in 0..nb_test {
                        // fill keys with random
                        Tensor::uniform_random_default(&mut rlwe_sk);
                        Tensor::uniform_random_default(&mut lwe_sk);

                        // allocation for the bootstrapping key
                        let mut trgsw: Vec<CTorus> = vec![
                            CTorus::zero();
                            cross::get_bootstrapping_key_size(
                                dimension,
                                polynomial_size,
                                level,
                                lwe_sk_n_bits
                            )
                        ];

                        // fill with the bootstrapping key already in the fourier domain
                        RGSW::create_fourier_bootstrapping_key(
                            &mut trgsw,
                            base_log,
                            level,
                            dimension,
                            polynomial_size,
                            std,
                            &lwe_sk,
                            &rlwe_sk,
                        );

                        // Create a fix message (encoded in the most significant bit of the torus)
                        // put a 3 bit message XXX here 0XXX000...000 in the torus bit representation
                        let val: Torus = ((polynomial_size as f64
                            - (5. * f64::sqrt(npe::cross::drift_index_lut(lwe_dimension))))
                            * (1. / (2. * polynomial_size as f64))
                            * (<Torus as Types>::TORUS_MAX as f64 + 1.))
                            as Torus;
                        let m0: Vec<Torus> = vec![val];

                        // allocate ciphertext vectors
                        let mut lwe_in: Vec<Torus> = vec![0; lwe_dimension + 1];
                        let mut lwe_out: Vec<Torus> = vec![0; dimension * polynomial_size + 1];

                        LWE::sk_encrypt(&mut lwe_in, &lwe_sk, &m0, lwe_dimension, std);
                        let cst = 1 << 29;
                        msg[i] = cst;
                        // create a constant accumulator
                        let mut accumulator: Vec<Torus> =
                            vec![0; (dimension + 1) * polynomial_size];
                        for i in 0..polynomial_size {
                            accumulator[dimension * polynomial_size + i] = cst;
                        }

                        // execute the bootstrapp
                        Cross::bootstrap(
                            &mut lwe_out,
                            &lwe_in,
                            &trgsw,
                            base_log,
                            level,
                            &mut accumulator,
                            polynomial_size,
                            dimension,
                        );

                        let mut m1: Vec<Torus> = vec![0];
                        // now the lwe is encrypted using a flatten of the trlwe encryption key
                        LWE::compute_phase(
                            &mut m1,
                            &rlwe_sk,
                            &lwe_out,
                            dimension * polynomial_size,
                        );
                        // store the decryption of the bootstrapped ciphertext
                        new_msg[i] = m1[0];
                    }

                    // call the NPE to find the theoretical amount of noise after the bootstrap
                    let output_variance = <$T as npe::Cross>::bootstrap(
                        lwe_dimension,
                        dimension,
                        level,
                        base_log,
                        polynomial_size,
                        f64::powi(std, 2),
                    );
                    // if we have enough test, we check that the obtain distribution is the same
                    // as the theoretical one
                    // if not, it only tests if the noise remains in the 99% confidence interval
                    if nb_test < 7 {
                        assert_delta_std_dev!(msg, new_msg, f64::sqrt(output_variance));
                    } else {
                        assert_noise_distribution!(
                            msg,
                            new_msg,
                            output_variance,
                            &format!("bootstrap-degree={}", polynomial_size)
                        );
                    }
                }
            }

            #[test]
            fn test_bootstrapp_drift() {
                // define settings
                let nb_test: usize = 10;
                let polynomial_size: usize = 1024;
                let dimension: usize = 1;
                let lwe_dimension: usize = 630;
                let n_slots = 1;
                let level = 3;
                let base_log = 7;
                let std = f64::powi(2., -29);
                let lwe_sk_n_bits = lwe_dimension; // as many bit as the length of th e lwe
                let log_degree = f64::log2(polynomial_size as f64) as i32;

                // compute the length of the lwe and rlwe secret keys
                let lwe_sk_len: usize =
                    <Torus as SecretKey>::get_secret_key_length(lwe_dimension, 1);
                let rlwe_sk_len: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);

                // allocate secret keys
                let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                let mut lwe_sk: Vec<Torus> = vec![0; lwe_sk_len];
                let mut msg: Vec<Torus> = vec![0; nb_test];
                let mut new_msg: Vec<Torus> = vec![0; nb_test];

                // launch nb_test tests
                for i in 0..nb_test {
                    // fill keys with random
                    Tensor::uniform_random_default(&mut rlwe_sk);
                    Tensor::uniform_random_default(&mut lwe_sk);

                    // allocation for the bootstrapping key
                    let mut trgsw: Vec<CTorus> = vec![
                        CTorus::zero();
                        cross::get_bootstrapping_key_size(
                            dimension,
                            polynomial_size,
                            level,
                            lwe_sk_n_bits
                        )
                    ];

                    // fill with the bootstrapping key already in the fourier domain
                    RGSW::create_fourier_bootstrapping_key(
                        &mut trgsw,
                        base_log,
                        level,
                        dimension,
                        polynomial_size,
                        std,
                        &lwe_sk,
                        &rlwe_sk,
                    );
                    let val: Torus = ((polynomial_size as f64
                        - (10. * f64::sqrt(npe::cross::drift_index_lut(lwe_dimension))))
                        * f64::powi(2., <Torus as Types>::TORUS_BIT as i32 - log_degree - 1))
                        as Torus;

                    let m0: Vec<Torus> = vec![val];
                    msg[i] = val;
                    let mut lwe_in: Vec<Torus> = vec![0; lwe_dimension + 1];
                    LWE::sk_encrypt(&mut lwe_in, &lwe_sk, &m0, lwe_dimension, std);

                    // accumulator is a trivial encryption of [0, 1/2N, 2/2N, ...]
                    let mut accumulator: Vec<Torus> =
                        vec![0; n_slots * (dimension + 1) * polynomial_size];
                    for i in 0..polynomial_size {
                        let val: Torus = (i as f64
                            * f64::powi(2., <Torus as Types>::TORUS_BIT as i32 - log_degree - 1))
                            as Torus;
                        accumulator[dimension * polynomial_size + i] = val as Torus;
                    }

                    // bootstrapp
                    let mut lwe_out: Vec<Torus> = vec![0; dimension * polynomial_size + 1];
                    Cross::bootstrap(
                        &mut lwe_out,
                        &lwe_in,
                        &trgsw,
                        base_log,
                        level,
                        &mut accumulator,
                        polynomial_size,
                        dimension,
                    );

                    // now the lwe is encrypted using a flatten of the trlwe encryption key
                    let mut m1: Vec<Torus> = vec![0];
                    LWE::compute_phase(&mut m1, &rlwe_sk, &lwe_out, dimension * polynomial_size);
                    new_msg[i] = m1[0];

                    // test that the drift remains within the bound of the theretical drift
                    let delta_max: i64 = ((5.
                        * f64::sqrt(npe::cross::drift_index_lut(lwe_dimension)))
                        * f64::powi(2., <Torus as Types>::TORUS_BIT as i32 - log_degree - 1))
                        as i64;
                    assert_delta_scalar!(m0[0], m1[0], delta_max);
                }
            }

            #[test]
            pub fn test_external_product() {
                let n_tests = 10;

                for _n in 0..n_tests {
                    // fix different polynomial degrees
                    let degrees = vec![512, 1024, 2048];
                    for polynomial_size in degrees {
                        // fix a set of parameters
                        let dimension: usize = 1;
                        let n_slots = 1;
                        let level = 6;
                        let base_log = 4;
                        let std_dev_bsk = f64::powi(2., -25);
                        let std_dev_rlwe = f64::powi(2., -20);

                        // compute the length of rlwe secret key
                        let rlwe_sk_len: usize =
                            <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                        let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                        Tensor::uniform_random_default(&mut rlwe_sk);

                        // we create a lwe secret key with one bit set to one
                        let lwe_sk: Vec<Torus> = vec![1 << (<Torus as Types>::TORUS_BIT - 1)];

                        // create the polynomial to encrypt
                        let mut messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
                        Tensor::uniform_random_default(&mut messages);

                        // allocate space for the decrypted polynomial
                        let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];

                        // allocation for the bootstrapping key
                        let mut trgsw: Vec<CTorus> = vec![
                            CTorus::zero();
                            cross::get_bootstrapping_key_size(
                                dimension,
                                polynomial_size,
                                level,
                                1
                            )
                        ];

                        RGSW::create_fourier_bootstrapping_key(
                            &mut trgsw,
                            base_log,
                            level,
                            dimension,
                            polynomial_size,
                            std_dev_bsk,
                            &lwe_sk,
                            &rlwe_sk,
                        );

                        // allocate vectors for rlwe ciphertexts (inputs)
                        let mut ciphertexts: Vec<Torus> =
                            vec![0; n_slots * (dimension + 1) * polynomial_size];
                        // let mut body_ciphertexts: Vec<Torus> = vec![0; n_slots * polynomial_size];

                        // allocate vectors for rlwe ciphertexts (output)
                        let mut res: Vec<Torus> =
                            vec![0; n_slots * (dimension + 1) * polynomial_size];
                        // let mut body_res: Vec<Torus> = vec![0; n_slots * polynomial_size];

                        // encrypt the polynomial
                        RLWE::sk_encrypt(
                            &mut ciphertexts,
                            &rlwe_sk,
                            &messages,
                            dimension,
                            polynomial_size,
                            std_dev_rlwe,
                        );

                        // unroll FFT Plan using FFTW
                        let mut fft: C2CPlanTorus = C2CPlanTorus::aligned(
                            &[polynomial_size],
                            Sign::Forward,
                            Flag::Measure,
                        )
                        .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
                        let mut ifft: C2CPlanTorus = C2CPlanTorus::aligned(
                            &[polynomial_size],
                            Sign::Backward,
                            Flag::Measure,
                        )
                        .expect("test_external_product: C2CPlanTorus::aligned threw an error...");

                        // allocate vectors used as temporary variables inside the external product
                        let mut mask_dec_i_fft: AlignedVec<CTorus> =
                            AlignedVec::new(polynomial_size);
                        let mut body_dec_i_fft: AlignedVec<CTorus> =
                            AlignedVec::new(polynomial_size);
                        let mut mask_res_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                        let mut body_res_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                        Cross::external_product_inplace_one_dimension(
                            &mut res,
                            &trgsw,
                            &mut ciphertexts,
                            polynomial_size,
                            base_log,
                            level,
                            &mut fft,
                            &mut ifft,
                            &mut mask_dec_i_fft,
                            &mut body_dec_i_fft,
                            &mut mask_res_fft,
                            &mut body_res_fft,
                        );
                        RLWE::compute_phase(
                            &mut new_messages,
                            &rlwe_sk,
                            &res,
                            dimension,
                            polynomial_size,
                        );

                        // call the NPE to find the theoritical amount of noise after the external product
                        let var_trgsw = std_dev_bsk * std_dev_bsk;
                        let var_trlwe = std_dev_rlwe * std_dev_rlwe;
                        let output_variance = <$T as npe::Cross>::external_product(
                            dimension,
                            level,
                            base_log,
                            polynomial_size,
                            var_trgsw,
                            var_trlwe,
                        );

                        // test
                        assert_noise_distribution!(
                            new_messages,
                            messages,
                            output_variance,
                            &format!("externalproduct-degree={}", polynomial_size)
                        );
                    }
                }
            }

            #[test]
            pub fn test_cmux_0() {
                // fix different polynomial degrees
                let degrees = vec![512, 1024, 2048];
                for polynomial_size in degrees {
                    // fix a set of parameters
                    let dimension: usize = 1;
                    let n_slots = 1;
                    let level = 4;
                    let base_log = 7;
                    let std_dev_bsk = f64::powi(2., -20);
                    let std_dev_rlwe = f64::powi(2., -25);

                    // compute the length of the rlwe secret key
                    let rlwe_sk_len: usize =
                        <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                    let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                    Tensor::uniform_random_default(&mut rlwe_sk);
                    let lwe_sk = vec![0];

                    // draw two random torus polynomials
                    let mut m0: Vec<Torus> = vec![0; n_slots * polynomial_size];
                    Tensor::uniform_random_default(&mut m0);
                    let mut m1: Vec<Torus> = vec![0; n_slots * polynomial_size];
                    Tensor::uniform_random_default(&mut m1);

                    // allocation for the decrypted result
                    let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];

                    // allocate and create the bootstrapping key
                    let mut trgsw: Vec<CTorus> =
                        vec![
                            CTorus::zero();
                            cross::get_bootstrapping_key_size(dimension, polynomial_size, level, 1)
                        ];

                    RGSW::create_fourier_bootstrapping_key(
                        &mut trgsw,
                        base_log,
                        level,
                        dimension,
                        polynomial_size,
                        std_dev_bsk,
                        &lwe_sk,
                        &rlwe_sk,
                    );

                    // allocate rlwe vectors
                    let mut ciphertexts0: Vec<Torus> =
                        vec![0; n_slots * (dimension + 1) * polynomial_size];
                    let mut ciphertexts1: Vec<Torus> =
                        vec![0; n_slots * (dimension + 1) * polynomial_size];

                    // encrypt polynomials
                    RLWE::sk_encrypt(
                        &mut ciphertexts0,
                        &rlwe_sk,
                        &m0,
                        dimension,
                        polynomial_size,
                        std_dev_rlwe,
                    );
                    RLWE::sk_encrypt(
                        &mut ciphertexts1,
                        &rlwe_sk,
                        &m1,
                        dimension,
                        polynomial_size,
                        std_dev_rlwe,
                    );

                    // unroll FFT Plan using FFTW
                    let mut fft: C2CPlanTorus =
                        C2CPlanTorus::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
                            .expect("test_cmux_0: C2CPlanTorus::aligned threw an error...");
                    let mut ifft: C2CPlanTorus =
                        C2CPlanTorus::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
                            .expect("test_cmux_0: C2CPlanTorus::aligned threw an error...");

                    // allocate vectors used as temporary variables inside the external product
                    let mut mask_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                    let mut body_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                    let mut res_fft: Vec<AlignedVec<CTorus>> =
                        vec![AlignedVec::new(polynomial_size); dimension + 1];

                    // compute cmux
                    Cross::cmux_inplace(
                        &mut ciphertexts0,
                        &mut ciphertexts1,
                        &trgsw,
                        polynomial_size,
                        dimension,
                        base_log,
                        level,
                        &mut fft,
                        &mut ifft,
                        &mut mask_dec_i_fft,
                        &mut body_dec_i_fft,
                        &mut res_fft,
                    );

                    // decrypt rlwe ciphertext
                    RLWE::compute_phase(
                        &mut new_messages,
                        &rlwe_sk,
                        &ciphertexts0,
                        dimension,
                        polynomial_size,
                    );

                    // call the NPE to find the theoretical amount of noise added by the cmux
                    let variance_rlwe = std_dev_rlwe * std_dev_rlwe;
                    let variance_trgsw = std_dev_bsk * std_dev_bsk;
                    let output_variance = <$T as npe::Cross>::cmux(
                        variance_rlwe,
                        variance_rlwe,
                        variance_trgsw,
                        dimension,
                        polynomial_size,
                        base_log,
                        level,
                    );

                    // test
                    println!("sqrt(max_var) = {}", f64::sqrt(output_variance));
                    for (elt0, elt1) in m0.iter().zip(new_messages.iter()) {
                        println!("{} - {}", *elt0, *elt1);
                    }
                    assert_noise_distribution!(
                        new_messages,
                        m0,
                        output_variance,
                        &format!("cmux0-degree={}", polynomial_size)
                    );
                }
            }

            #[test]
            pub fn test_cmux_1() {
                // fix different polynomial degrees
                let degrees = vec![512, 1024, 2048];
                for polynomial_size in degrees {
                    // fix a set of parameters
                    let dimension: usize = 1;
                    let n_slots = 1;
                    let level = 4;
                    let base_log = 7;
                    let std_dev_bsk = f64::powi(2., -20);
                    let std_dev_rlwe = f64::powi(2., -25);

                    // compute the length of the rlwe secret key
                    let rlwe_sk_len: usize =
                        <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                    let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                    Tensor::uniform_random_default(&mut rlwe_sk);
                    let lwe_sk = vec![1 << (<Torus as Types>::TORUS_BIT - 1)];

                    // draw two random torus polynomials
                    let mut m0: Vec<Torus> = vec![0; n_slots * polynomial_size];
                    Tensor::uniform_random_default(&mut m0);
                    let mut m1: Vec<Torus> = vec![0; n_slots * polynomial_size];
                    Tensor::uniform_random_default(&mut m1);

                    // allocation for the decrypted result
                    let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];

                    // allocate and create the bootstrapping key
                    let mut trgsw: Vec<CTorus> =
                        vec![
                            CTorus::zero();
                            cross::get_bootstrapping_key_size(dimension, polynomial_size, level, 1)
                        ];

                    RGSW::create_fourier_bootstrapping_key(
                        &mut trgsw,
                        base_log,
                        level,
                        dimension,
                        polynomial_size,
                        std_dev_bsk,
                        &lwe_sk,
                        &rlwe_sk,
                    );

                    // allocate rlwe vectors
                    let mut ciphertexts0: Vec<Torus> =
                        vec![0; n_slots * (dimension + 1) * polynomial_size];
                    let mut ciphertexts1: Vec<Torus> =
                        vec![0; n_slots * (dimension + 1) * polynomial_size];

                    // encrypt polynomials
                    RLWE::sk_encrypt(
                        &mut ciphertexts0,
                        &rlwe_sk,
                        &m0,
                        dimension,
                        polynomial_size,
                        std_dev_rlwe,
                    );
                    RLWE::sk_encrypt(
                        &mut ciphertexts1,
                        &rlwe_sk,
                        &m1,
                        dimension,
                        polynomial_size,
                        std_dev_rlwe,
                    );

                    // unroll FFT Plan using FFTW
                    let mut fft: C2CPlanTorus =
                        C2CPlan::aligned(&[polynomial_size], Sign::Forward, Flag::Measure)
                            .expect("test_cmux_1: C2CPlanTorus::aligned threw an error...");
                    let mut ifft: C2CPlanTorus =
                        C2CPlan::aligned(&[polynomial_size], Sign::Backward, Flag::Measure)
                            .expect("test_cmux_1: C2CPlanTorus::aligned threw an error...");

                    // allocate vectors used as temporary variables inside the external product
                    let mut mask_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                    let mut body_dec_i_fft: AlignedVec<CTorus> = AlignedVec::new(polynomial_size);
                    let mut res_fft: Vec<AlignedVec<CTorus>> =
                        vec![AlignedVec::new(polynomial_size); dimension + 1];

                    // compute cmux
                    Cross::cmux_inplace(
                        &mut ciphertexts0,
                        &mut ciphertexts1,
                        &trgsw,
                        polynomial_size,
                        dimension,
                        base_log,
                        level,
                        &mut fft,
                        &mut ifft,
                        &mut mask_dec_i_fft,
                        &mut body_dec_i_fft,
                        &mut res_fft,
                    );

                    // decrypt rlwe ciphertext
                    RLWE::compute_phase(
                        &mut new_messages,
                        &rlwe_sk,
                        &ciphertexts0,
                        dimension,
                        polynomial_size,
                    );

                    // call the NPE to find the theoretical amount of noise added by the cmux
                    let variance_rlwe = std_dev_rlwe * std_dev_rlwe;
                    let variance_trgsw = std_dev_bsk * std_dev_bsk;
                    let output_variance = <$T as npe::Cross>::cmux(
                        variance_rlwe,
                        variance_rlwe,
                        variance_trgsw,
                        dimension,
                        polynomial_size,
                        base_log,
                        level,
                    );

                    // test
                    assert_noise_distribution!(
                        new_messages,
                        m1,
                        output_variance,
                        &format!("cmux1-degree={}", polynomial_size)
                    );
                }
            }

            #[test]
            fn test_sample_extract() {
                let n_tests = N_TESTS;
                // fix different polynomial degrees
                let degrees = vec![512, 1024, 2048];
                for polynomial_size in degrees {
                    // fixa set of parameters
                    let mut sdk_samples: Vec<Torus> = vec![0; n_tests];
                    let mut groundtruth_samples: Vec<Torus> = vec![0; n_tests];
                    let std_dev = f64::powi(2., -20);
                    let n_slots: usize = 1;
                    let dimension = 1;

                    // compute length of the lwe secret key
                    let sk_len: usize =
                        <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                    let mut sk: Vec<Torus> = vec![0; sk_len];
                    for i in 0..n_tests {
                        // fill the secret key with random
                        Tensor::uniform_random_default(&mut sk);

                        // allocate and draw a random polynomial
                        let mut messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
                        Tensor::uniform_random_default(&mut messages);

                        // allocate RLWE ciphertext
                        let mut ciphertexts: Vec<Torus> =
                            vec![0; n_slots * (dimension + 1) * polynomial_size];
                        // encrypt RLWE sample
                        RLWE::sk_encrypt(
                            &mut ciphertexts,
                            &sk,
                            &messages,
                            dimension,
                            polynomial_size,
                            std_dev,
                        );

                        // allocate LWE ciphertext
                        let mut lwe: Vec<Torus> =
                            vec![0; n_slots * dimension * polynomial_size + 1 * n_slots];

                        // allocate space of the decrypted message (after sample extract)
                        let mut new_messages: Vec<Torus> = vec![0; 1];

                        // perform sample extract
                        Cross::constant_sample_extract(
                            &mut lwe,
                            &ciphertexts,
                            dimension,
                            polynomial_size,
                        );

                        // decrypt resulting lwe ciphertext
                        LWE::compute_phase(
                            &mut new_messages,
                            &sk,
                            &lwe,
                            dimension * polynomial_size,
                        );
                        groundtruth_samples[i] = messages[0];
                        sdk_samples[i] = new_messages[0];
                    }

                    // test
                    if n_tests < 7 {
                        assert_delta_std_dev!(groundtruth_samples, sdk_samples, std_dev);
                    } else {
                        assert_noise_distribution!(
                            groundtruth_samples,
                            sdk_samples,
                            f64::powi(std_dev, 2),
                            &format!("sampleextract-degree={}", polynomial_size)
                        );
                    }
                }
            }

            #[test]
            pub fn test_external_product_generic() {
                let n_tests = 10;
                for _n in 0..n_tests {
                    // fix different polynomial degrees
                    let degrees = vec![512, 1024, 2048];
                    for polynomial_size in degrees {
                        // fix a set of parameters
                        let dimension: usize = 2;
                        let n_slots = 1;
                        let level = 6;
                        let base_log = 4;
                        let std_dev_bsk = f64::powi(2., -25);
                        let std_dev_rlwe = f64::powi(2., -20);

                        // compute the length of rlwe secret key
                        let rlwe_sk_len: usize =
                            <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                        let mut rlwe_sk: Vec<Torus> = vec![0; rlwe_sk_len];
                        Tensor::uniform_random_default(&mut rlwe_sk);

                        // we create a lwe secret key with one bit set to one
                        let lwe_sk: Vec<Torus> = vec![1 << (<Torus as Types>::TORUS_BIT - 1)];

                        // create the polynomial to encrypt
                        let mut messages: Vec<Torus> = vec![0; n_slots * polynomial_size];
                        Tensor::uniform_random_default(&mut messages);

                        // allocate space for the decrypted polynomial
                        let mut new_messages: Vec<Torus> = vec![0; n_slots * polynomial_size];

                        // allocation for the bootstrapping key
                        let mut trgsw: Vec<CTorus> = vec![
                            CTorus::zero();
                            cross::get_bootstrapping_key_size(
                                dimension,
                                polynomial_size,
                                level,
                                1
                            )
                        ];
                        RGSW::create_fourier_bootstrapping_key(
                            &mut trgsw,
                            base_log,
                            level,
                            dimension,
                            polynomial_size,
                            std_dev_bsk,
                            &lwe_sk,
                            &rlwe_sk,
                        );
                        // allocate vectors for rlwe ciphertexts (inputs)
                        let mut ciphertexts: Vec<Torus> =
                            vec![0; n_slots * (dimension + 1) * polynomial_size];

                        // allocate vectors for rlwe ciphertexts (outputs)
                        let mut res: Vec<Torus> =
                            vec![0; n_slots * (dimension + 1) * polynomial_size];

                        // encrypt the polynomial
                        RLWE::sk_encrypt(
                            &mut ciphertexts,
                            &rlwe_sk,
                            &messages,
                            dimension,
                            polynomial_size,
                            std_dev_rlwe,
                        );

                        // unroll FFT Plan using FFTW
                        let mut fft: C2CPlanTorus = C2CPlanTorus::aligned(
                            &[polynomial_size],
                            Sign::Forward,
                            Flag::Measure,
                        )
                        .expect("test_external_product: C2CPlanTorus::aligned threw an error...");
                        let mut ifft: C2CPlanTorus = C2CPlanTorus::aligned(
                            &[polynomial_size],
                            Sign::Backward,
                            Flag::Measure,
                        )
                        .expect("test_external_product: C2CPlanTorus::aligned threw an error...");

                        // allocate vectors used as temporary variables inside the external product
                        let mut mask_dec_i_fft: AlignedVec<CTorus> =
                            AlignedVec::new(polynomial_size);
                        let mut body_dec_i_fft: AlignedVec<CTorus> =
                            AlignedVec::new(polynomial_size);
                        let mut res_fft: Vec<AlignedVec<CTorus>> =
                            vec![AlignedVec::new(polynomial_size); dimension + 1];
                        Cross::external_product_inplace(
                            &mut res,
                            &trgsw,
                            &mut ciphertexts,
                            polynomial_size,
                            dimension,
                            base_log,
                            level,
                            &mut fft,
                            &mut ifft,
                            &mut mask_dec_i_fft,
                            &mut body_dec_i_fft,
                            &mut res_fft,
                        );
                        RLWE::compute_phase(
                            &mut new_messages,
                            &rlwe_sk,
                            &res,
                            dimension,
                            polynomial_size,
                        );

                        // call the NPE to find the theoritical amount of noise after the external product
                        let var_trgsw = std_dev_bsk * std_dev_bsk;
                        let var_trlwe = std_dev_rlwe * std_dev_rlwe;
                        let output_variance = <$T as npe::Cross>::external_product(
                            dimension,
                            level,
                            base_log,
                            polynomial_size,
                            var_trgsw,
                            var_trlwe,
                        );

                        // test
                        assert_noise_distribution!(
                            new_messages,
                            messages,
                            output_variance,
                            &format!("externalproductgeneric-degree={}", polynomial_size)
                        );
                    }
                }
            }
        }
    };
}

cross_test_mod!(u32, tests_u32);
cross_test_mod!(u64, tests_u64);
