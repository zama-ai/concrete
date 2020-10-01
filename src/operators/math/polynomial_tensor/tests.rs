macro_rules! polynomial_tensor_test_mod {
    ($T:ty,$MN:ident) => {
        mod $MN {
            use crate::operators::crypto::SecretKey;
            use crate::operators::math::PolynomialTensor;
            use crate::operators::math::Tensor;
            use rand;
            use rand::Rng;

            type Torus = $T;

            #[test]
            fn test_compute_binary_multisum() {
                //! only tests if compute_binary_multisum, compute_binary_multisum_monome,
                //! add_binary_multisum and sub_binary_multisum gives the same results
                // settings
                let polynomial_size: usize = 128;
                let dimension: usize = 4;
                let size: usize = dimension * polynomial_size;

                // generates a random Torus polynomial tensor
                let mut t_torus: Vec<Torus> = vec![0; size];
                Tensor::uniform_random_default(&mut t_torus);

                // generates a random boolean key
                let key_size: usize =
                    <Torus as SecretKey>::get_secret_key_length(dimension, polynomial_size);
                let mut t_bool: Vec<Torus> = vec![0; key_size];
                Tensor::uniform_random_default(&mut t_bool);

                // compute_binary_multisum
                let mut res_0: Vec<Torus> = vec![0; polynomial_size];
                Tensor::uniform_random_default(&mut res_0);
                PolynomialTensor::compute_binary_multisum(
                    &mut res_0,
                    &t_torus,
                    &t_bool,
                    polynomial_size,
                );

                // compute_binary_multisum_monome
                let mut res_1: Vec<Torus> = vec![0; polynomial_size];
                for (i, monome) in res_1.iter_mut().enumerate() {
                    *monome = PolynomialTensor::compute_binary_multisum_monome(
                        &t_torus,
                        &t_bool,
                        polynomial_size,
                        dimension,
                        i,
                    );
                }

                // add_binary_multisum
                let mut res_2: Vec<Torus> = vec![0; polynomial_size];
                PolynomialTensor::add_binary_multisum(
                    &mut res_2,
                    &t_torus,
                    &t_bool,
                    polynomial_size,
                );

                // sub_binary_multisum
                let mut res_3: Vec<Torus> = vec![0; polynomial_size];
                PolynomialTensor::sub_binary_multisum(
                    &mut res_3,
                    &t_torus,
                    &t_bool,
                    polynomial_size,
                );
                for el in res_3.iter_mut() {
                    *el = (0 as Torus).wrapping_sub(*el);
                }

                // tests
                assert_eq!(res_0, res_1);
                assert_eq!(res_0, res_2);
                assert_eq!(res_0, res_3);
            }

            #[test]
            fn test_multiply_divide_by_monomial() {
                //! tests if multiply_by_monomial and divide_by_monomial cancel each other
                let mut rng = rand::thread_rng();

                // settings
                let polynomial_size: usize = 1024;

                // generates a random Torus polynomial tensor
                let mut poly: Vec<Torus> = vec![0; polynomial_size];
                Tensor::uniform_random_default(&mut poly);

                // copy that tensor
                let ground_truth = poly.clone();

                // generates a random r
                let mut r: usize = rng.gen();
                r = r % polynomial_size;

                // multiply by X^r and then divides by X^r
                PolynomialTensor::multiply_by_monomial(&mut poly, r, polynomial_size);
                PolynomialTensor::divide_by_monomial(&mut poly, r, polynomial_size);

                // test
                assert_eq!(&poly, &ground_truth);

                // generates a random r_big
                let mut r_big: usize = rng.gen();
                r_big = r_big % polynomial_size + 2000;

                // multiply by X^r_big and then divides by X^r_big
                PolynomialTensor::multiply_by_monomial(&mut poly, r_big, polynomial_size);
                PolynomialTensor::divide_by_monomial(&mut poly, r_big, polynomial_size);

                // test
                assert_eq!(&poly, &ground_truth);

                // divides by X^r_big and then multiply by X^r_big
                PolynomialTensor::divide_by_monomial(&mut poly, r_big, polynomial_size);
                PolynomialTensor::multiply_by_monomial(&mut poly, r_big, polynomial_size);

                // test
                assert_eq!(&poly, &ground_truth);
            }
        }
    };
}

polynomial_tensor_test_mod!(u32, tests_u32);
polynomial_tensor_test_mod!(u64, tests_u64);
