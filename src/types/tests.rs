macro_rules! types_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::operators::math::Random;
            use crate::types::FTorus;
            use crate::Types;
            use itertools::enumerate;
            use rand::Rng;
            use std::panic;

            type Torus = $T;

            #[test]
            fn test_round_to_closest_multiple() {
                //! test with 30 random Torus elements already rounded that if we add / subtract them a small value, when rounded they are still the same original value
                let nb_test: usize = 30; // number of tested values
                let mut rng = rand::thread_rng();
                let mut log_b: usize = rng.gen();
                let mut level_max: usize = rng.gen();
                log_b = (log_b % ((<Torus as Types>::TORUS_BIT / 4) - 1)) + 1;
                level_max = (level_max % 4) + 1;
                let nb_bit: usize = log_b * level_max;

                // generate random values that will be rounded after adding a small delta
                let mut values: Vec<Torus> = vec![0; nb_test];
                Random::rng_uniform_n_msb(&mut values, nb_bit);

                // generate deltas
                let mut deltas: Vec<Torus> = vec![0; nb_test];
                Random::rng_uniform_n_lsb(&mut deltas, <Torus as Types>::TORUS_BIT - nb_bit - 1); // delta is inferior to half the smallest non-zero rounded possible

                // when adding delta
                let mut res: Vec<Torus> = vec![0; nb_test];
                for i in 0..nb_test {
                    res[i] = Types::round_to_closest_multiple(
                        values[i].wrapping_add(deltas[i]),
                        log_b,
                        level_max,
                    );
                }
                assert_eq!(values, res);

                // when subtracting delta
                for i in 0..nb_test {
                    res[i] = Types::round_to_closest_multiple(
                        values[i].wrapping_sub(deltas[i]),
                        log_b,
                        level_max,
                    );
                }
                assert_eq!(values, res);
            }

            #[test]
            #[cfg(debug_assertions)]
            fn test_panic_round_to_closest_multiple() {
                //! test that it panics when log_b * level_max==TORUS_BIT

                let mut rng = rand::thread_rng();
                let log_b: usize = <Torus as Types>::TORUS_BIT / 4;
                let level_max: usize = 4;

                // generate a random value that will be rounded
                let value: Torus = rng.gen();

                let result = panic::catch_unwind(|| {
                    Types::round_to_closest_multiple(value, log_b, level_max);
                });

                // test
                assert!(result.is_err());
            }

            #[test]
            fn test_torus_small_sign_decompose() {
                //! generate a random Torus element, round it, computes its signed decomposition and recompose it to test the equality with the rounded element

                let mut rng = rand::thread_rng();

                // generate random settings
                let mut log_b: usize = rng.gen();
                let mut level_max: usize = rng.gen();
                log_b = (log_b % (<Torus as Types>::TORUS_BIT / 4)) + 1;
                level_max = (level_max % 4) + 1;

                // generate a random Torus element x and round it
                let mut x: Torus = rng.gen();
                // let mut x: Torus = rng.gen::<Torus>().rem_euclid(1 << 30);

                if <Torus as Types>::TORUS_BIT > level_max * log_b {
                    // round x to keep only the bits used it the signed decomposition
                    x = Types::round_to_closest_multiple(x, log_b, level_max);
                }

                // decompose the rounded x
                let mut decomp_x: Vec<Torus> = vec![0; level_max];
                Types::torus_small_sign_decompose(&mut decomp_x, x, log_b);

                // recompose the Torus element
                let mut recomp_x: i64 = 0;
                for (i, di) in enumerate(decomp_x.iter()) {
                    recomp_x = recomp_x
                        + ((*di as <Torus as Types>::STorus) as i64)
                            * (Types::set_val_at_level_l(1 as Torus, log_b, i) as i64);
                }
                let recomp_res: Torus;
                // deal with the negative sign of the recomposition
                if recomp_x < 0 {
                    let tmp: f64 =
                        f64::powi(2.0, <Torus as Types>::TORUS_BIT as i32) + (recomp_x as f64);
                    recomp_res = tmp as Torus;
                } else {
                    recomp_res = recomp_x as Torus;
                }
                // assert the equality between the rounded element and its recomposition
                assert_eq!(recomp_res, x);
            }

            #[test]
            fn test_signed_decompose_one_level() {
                //! This test picks 1000 random Torus values,
                //! rounds them according to the decomposition precision (base_log*level_max) which is randomly picked each time,
                //! decomposes them with the signed_decompose_one_level() function,
                //! recomposes Torus elements,
                //! and finally makes sure that they are equal to the rounded values.
                let n_tests = 1000;
                for _i in 0..n_tests {
                    let mut rng = rand::thread_rng();

                    // generate random settings
                    let mut base_log: usize = rng.gen();
                    let mut level_max: usize = rng.gen();
                    base_log = (base_log % ((<Torus as Types>::TORUS_BIT / 4) - 1)) + 1;
                    level_max = (level_max % 4) + 1;
                    println!("logB:{}, levelMax:{}", base_log, level_max);

                    // generate a random x Torus element and round it
                    let mut x: Torus = rng.gen();
                    x = Types::round_to_closest_multiple(x, base_log, level_max);
                    println!(
                        "x:{} -> {}",
                        x,
                        Types::torus_binary_representation(x, base_log)
                    );

                    // decompose the rounded x
                    let mut decomp_x: Vec<Torus> = vec![0; level_max];
                    let mut carries: Vec<Torus> = vec![0; level_max];
                    for i in (0..level_max).rev() {
                        let pair = Types::signed_decompose_one_level(x, carries[i], base_log, i);
                        decomp_x[i] = pair.0;
                        if i > 0 {
                            carries[i - 1] = pair.1;
                        }
                        println!("XXdecomp_{} -> {}", i, decomp_x[i]);
                    }

                    // recompose the Torus element
                    let mut recomp_x: Torus = 0;
                    for (i, di) in enumerate(decomp_x.iter()) {
                        println!("decomp_{} -> {}", i, di);
                        let mut tmp: FTorus = ((*di as <Torus as Types>::STorus) as FTorus)
                            * (Types::set_val_at_level_l(1 as Torus, base_log, i) as FTorus);
                        if tmp < 0. {
                            tmp = -tmp;
                            recomp_x = recomp_x.wrapping_sub(tmp as Torus);
                        } else {
                            recomp_x = recomp_x.wrapping_add(tmp as Torus);
                        }
                    }

                    println!(
                        "recomp x:{} -> {}",
                        recomp_x,
                        Types::torus_binary_representation(recomp_x, base_log)
                    );

                    println!("recomp:{} -> x:{}\n", recomp_x, x);

                    // test
                    assert_eq!(recomp_x, x);
                }
            }
        }
    };
}

types_test_mod!(u32, tests_u32);
types_test_mod!(u64, tests_u64);
