macro_rules! random_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::operators::math::Random;
            use crate::operators::math::Tensor;

            type Torus = $T;

            #[test]
            fn test_normal_random() {
                //! test if the normal random generation with std_dev is below 3*std_dev (99.7%)

                // settings
                let std_dev: f64 = f64::powi(2., -20);
                let k = 10;

                // generates normal random
                let mut samples_int: Vec<Torus> = vec![0; k];
                // Random::vectorial_rng_normal(&mut samples_int, 0., std_dev);
                Random::vectorial_openssl_normal(&mut samples_int, 0., std_dev);
                for elt in samples_int[..10].iter() {
                    println!("{}", elt);
                }

                // converts into float
                let mut samples_float: Vec<f64> = vec![0.; k];
                Tensor::int_to_float(&mut samples_float, &samples_int);
                for x in samples_float.iter_mut() {
                    if *x > 0.5 {
                        *x = 1. - *x;
                    }
                }

                // tests if over 3*std_dev
                let mut number_of_samples_outside_confidence_interval: Torus = 0;
                for s in samples_float.iter() {
                    if *s > 3. * std_dev || *s < -3. * std_dev {
                        number_of_samples_outside_confidence_interval += 1;
                    }
                }

                // computes the percentage of samples over 3*std_dev
                let proportion_of_samples_outside_confidence_interval: f64 =
                    (number_of_samples_outside_confidence_interval as f64) / (k as f64);

                for elt in samples_float[..10].iter() {
                    println!("{}", elt);
                }

                // test
                assert!(
                    proportion_of_samples_outside_confidence_interval < 0.003,
                    "test normal random : proportion = {} ; n = {}",
                    proportion_of_samples_outside_confidence_interval,
                    number_of_samples_outside_confidence_interval
                );

                // panic!();
            }
            #[test]
            fn test_distribution() {
                // settings
                let std_dev: f64 = f64::powi(2., -5);
                let k = 10000;

                // generates normal random
                let mut openssl_samples: Vec<Torus> = vec![0; k];
                let veczeros = vec![0; k];
                // Random::vectorial_rng_normal(&mut samples_int, 0., std_dev);
                Random::vectorial_openssl_normal(&mut openssl_samples, 0., std_dev);

                assert_noise_distribution!(
                    veczeros,
                    openssl_samples,
                    f64::powi(std_dev, 2),
                    "normal"
                );
            }
        }
    };
}

random_test_mod!(u32, tests_u32);
random_test_mod!(u64, tests_u64);
