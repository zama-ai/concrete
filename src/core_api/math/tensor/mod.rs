//! Tensor Operations
//! * Contains every function related to generic Torus tensors, but also a few functions for CTorus and FTorus tensors
//! * In this file there is no functions that would have a different implementation depending on the type of the manipulated ciphertext i.e. LWE, RLWE or GSW for instance

#[cfg(test)]
mod tests;

use crate::core_api::crypto::SecretKey;
use crate::core_api::math::Random;
use crate::types::{CTorus, FTorus, FTORUS_BIT};
use crate::Types;
use itertools::izip;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::mem::transmute;

pub trait Tensor: Sized {
    fn uniform_random_default(res: &mut [Self]);
    fn uniform_random_with_some_zeros(res: &mut [Self], probability: f64);
    fn normal_random_default(res: &mut [Self], mean: f64, std_dev: f64);
    fn get_normal_random_default(mean: f64, std_dev: f64) -> Self;
    fn add(res: &mut [Self], t0: &[Self], t1: &[Self]);
    fn add_inplace(res: &mut [Self], t: &[Self]);
    fn neg_inplace(res: &mut [Self]);

    fn sub(res: &mut [Self], t0: &[Self], t1: &[Self]);
    fn sub_inplace(res: &mut [Self], t: &[Self]);
    fn scalar_mul(res: &mut [Self], t_in: &[Self], n: Self);
    fn scalar_mul_inplace(t_in: &mut [Self], n: Self);
    fn sub_scalar_mul(res: &mut [Self], t_in: &[Self], n: Self);
    fn add_scalar_mul(res: &mut [Self], t_in: &[Self], n: Self);
    fn binary_multisum(t_torus: &[Self], t_bool: &[Self]) -> Self;
    fn add_several_binary_multisum(t_res: &mut [Self], t_torus: &[Self], t_bool: &[Self]);
    fn get_binary_multisum(t_torus: &[Self], t_bool: &[Self]) -> Self;
    fn sub_several_binary_multisum(t_res: &mut [Self], t_torus: &[Self], t_bool: &[Self]);
    fn round_to_closest_multiple(
        t_res: &mut [Self],
        t_input: &[Self],
        base_log: usize,
        max_level: usize,
    );
    fn int_to_float(t_res: &mut [f64], t_int: &[Self]);
    fn float_to_int(t_res: &mut [Self], t_float: &[f64]);
    fn compute_modular_distance(t_res: &mut [Self], t_x: &[Self], t_y: &[Self]);
    fn compute_signed_modular_distance(t_res: &mut [f64], t_x: &[Self], t_y: &[Self]);
    fn print(tensor: &[Self]);
    fn write_in_file(tensor: &[Self], path: &str) -> std::io::Result<()>;
    fn read_in_file(tensor: &mut [Self], path: &str) -> std::io::Result<()>;
    fn to_string_torus_binary_representation(tensor: &[Self], base_log: usize) -> String;
    fn bit_shift_left_inplace(t_in: &mut [Self], nb: usize);
}

macro_rules! impl_trait_tensor {
    ($T:ty,$DOC:expr) => {
        impl Tensor for $T {
            /// Fill res with the opposite of res (in the torus)
            /// # Description
            /// res <- - res
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last two tensors)
            ///
            /// // neg
            /// Tensor::neg_inplace(&mut t);
            /// ```
            fn neg_inplace(res: &mut [$T]) {
                for res_ref in res.iter_mut() {
                    *res_ref = res_ref.wrapping_neg() ;
                }
            }

            /// Fills res with uniformly random values from the default source
            /// # Description
            /// res <- uniform random
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 6;
            ///
            /// // creation of a tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            /// Tensor::uniform_random_default(&mut t);
            /// ```
            fn uniform_random_default(res: &mut [$T]) {
                // Call to a PRNG
                #[cfg(all(feature = "unsafe"))]
                Random::rng_uniform(res) ;
                // Call to a CSPRNG
                #[cfg(not(any(feature = "unsafe")))]
                Random::openssl_uniform(res) ;
            }

            /// Either fills elements of res with uniform random or with a zero according to probability
            /// if probability is set to 0.2, then there will be approximately 1/5 random elements
            /// and the rest will be set to zero
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `probability` - the probability for an element to be uniformly random
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 15;
            ///
            /// // creation of a tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            /// Tensor::uniform_random_with_some_zeros(&mut t, 0.2);
            ///
            /// // [80 percents of zeros] tensor = [0, 0, 0, 0, 2085785791, 0, 0, 0, 607256796, 0, 195350706, 0, 0, 0, 0, ]
            /// ```
            fn uniform_random_with_some_zeros(res: &mut [$T], probability: f64) {
                Random::rng_uniform_with_some_zeros(
                    res,
                    (probability * f64::powi(2., <$T as Types>::TORUS_BIT as i32)) as $T,
                );
            }

            /// Fill res with normal random values
            /// # Description
            /// res <- normal_random(mean,std_dev)
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 6;
            ///
            /// // creation of a tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            /// Tensor::normal_random_default(&mut t, 0., 0.00003);
            /// ```
            fn normal_random_default(res: &mut [$T], mean: f64, std_dev: f64) {
                // Call to a PRNG
                #[cfg(all(feature = "unsafe"))]
                Random::vectorial_rng_normal(res, mean, std_dev);
                // Call to a CSPRNG
                #[cfg(not(any(feature = "unsafe")))]
                Random::vectorial_openssl_normal(res, mean, std_dev);
            }

            /// Fill res with normal random values
            /// # Arguments
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Output
            /// * return a sample drawn from a normal distribution with the given hyper-parameters(mean,std_dev)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 6;
            ///
            /// // draw a random sample
            /// let a: Torus = Tensor::get_normal_random_default(0., 0.00003);
            /// ```
            fn get_normal_random_default(mean: f64, std_dev: f64) -> $T {
                // Call to a PRNG
                #[cfg(all(feature = "unsafe"))]
                return Random::rng_normal(mean, std_dev);
                // Call to a CSPRNG
                #[cfg(not(any(feature = "unsafe")))]
                return Random::openssl_normal(mean, std_dev);
            }

            /// Fill res with the sum of t0 and t1 (in the torus)
            /// # Description
            /// res <- t0 + t1
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t0` - Torus slice
            /// * `t1` - Torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last two tensors)
            ///
            /// // allocation of a tensor for the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // add
            /// Tensor::add(&mut res, &t0, &t1);
            /// ```
            fn add(res: &mut [$T], t0: &[$T], t1: &[$T]) {
                debug_assert!(res.len() == t0.len(), "core_api::math::tensor::add : res.len() = {} != t0.len() = {}", res.len(), t0.len()) ;
                debug_assert!(res.len() == t1.len(), "core_api::math::tensor::add : res.len() = {} != t1.len() = {}", res.len(), t1.len()) ;

                for ((res_ref, t0_val), t1_val) in res.iter_mut().zip(t0).zip(t1) {
                    *res_ref = t0_val.wrapping_add(*t1_val);
                }
            }

            /// Fill res with the sum of res and t0 (in the torus)
            /// # Description
            /// res <- res + t
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t`- Torus sclice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last two tensors)
            ///
            /// // add
            /// Tensor::add_inplace(&mut t0, &t1);
            /// ```
            fn add_inplace(res: &mut [$T], t: &[$T]) {
                debug_assert!(res.len() == t.len(), "core_api::math::tensor::add_inplace : res.len() = {} != t.len() = {}", res.len(), t.len()) ;
                for (res_ref, t_val) in res.iter_mut().zip(t) {
                    *res_ref = res_ref.wrapping_add(*t_val);
                }
            }

            /// Fill res with t0 minus t1 (in the torus)
            /// # Description
            /// res <- t0 - t1
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t0` - Torus slice
            /// * `t1` - Torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last two tensors)
            ///
            /// // allocation of a tensor for the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // subtract
            /// Tensor::sub(&mut res, &t0, &t1);
            /// ```
            fn sub(res: &mut [$T], t0: &[$T], t1: &[$T]) {
                debug_assert!(res.len() == t0.len(), "core_api::math::tensor::sub : res.len() = {} != t0.len() = {}", res.len(), t0.len()) ;
                debug_assert!(res.len() == t1.len(), "core_api::math::tensor::sub : res.len() = {} != t1.len() = {}", res.len(), t1.len()) ;
                for ((res_ref, t0_val), t1_val) in res.iter_mut().zip(t0).zip(t1) {
                    *res_ref = t0_val.wrapping_sub(*t1_val);
                }
            }

            /// Fill res with res minus t0 (in the torus)
            /// # Description
            /// res <- res - t0
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t` - Torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last two tensors)
            ///
            /// // subtract
            /// Tensor::sub_inplace(&mut t0, &t1);
            /// ```
            fn sub_inplace(res: &mut [$T], t: &[$T]) {
                debug_assert!(res.len() == t.len(), "core_api::math::tensor::sub_inplace : res.len() = {} != t.len() = {}", res.len(), t.len()) ;

                for (res_ref, t_val) in res.iter_mut().zip(t) {
                    *res_ref = res_ref.wrapping_sub(*t_val);
                }
            }

            /// Multiply res by a scalar n
            /// # Description
            /// res <- tin * n
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t_in` - Torus slice
            /// * `n` - integer (u32 or u64)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // select a scalar
            /// let scal: Torus = 12;
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // scalar multiplication
            /// Tensor::scalar_mul(&mut res, &t, scal);
            /// ```
            fn scalar_mul(res: &mut [$T], t_in: &[$T], n: $T) {
                debug_assert!(res.len() == t_in.len(), "core_api::math::tensor::scalar_mul : res.len() = {} != t_in.len() = {}", res.len(), t_in.len()) ;

                for (res_ref, t_in_val) in res.iter_mut().zip(t_in) {
                    *res_ref = t_in_val.wrapping_mul(n);
                }
            }

            /// Multiply res by n, a signed integer
            /// # Description
            /// tin <- tin * n
            /// # Arguments
            /// * `t_in` - Torus slice (output)
            /// * `n` - integer (u32 or u64)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // select a scalar
            /// let scal: Torus = 12;
            ///
            /// // scalar multiplication
            /// Tensor::scalar_mul_inplace(&mut t, scal);
            /// ```
            fn scalar_mul_inplace(t_in: &mut [$T],  n: $T) {

                for  t_in_val in t_in.iter_mut() {
                    *t_in_val = t_in_val.wrapping_mul(n);
                }
            }

            /// Shift to the left res by n bits
            /// # Description
            /// tin <- tin << n
            /// # Arguments
            /// * `t_in` - Torus slice (output)
            /// * `n` - usize
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // select a scalar
            /// let nb: usize = 4;
            ///
            /// // scalar multiplication
            /// Tensor::bit_shift_left_inplace(&mut t, nb);
            /// ```
            fn bit_shift_left_inplace(t_in: &mut [$T],  n: usize) {
                for  t_in_val in t_in.iter_mut() {
                    *t_in_val = t_in_val.wrapping_shl(n as u32);
                }
            }

            /// Subtract to res the multiplication of t_in by a scalar n
            /// # Description
            /// res <- res - t_in * n
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t_in` - Torus slice
            /// * `n` - integer (u32 or u64)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // select a scalar
            /// let scal: Torus = 12;
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // subtract the scalar multiplication
            /// Tensor::sub_scalar_mul(&mut res, &t, scal);
            /// ```
            fn sub_scalar_mul(res: &mut [$T], t_in: &[$T], n: $T) {
                debug_assert!(res.len() == t_in.len(), "core_api::math::tensor::sub_scalar_mul : res.len() = {} != t_in.len() = {}", res.len(), t_in.len()) ;
                for (res_ref, t_in_val) in res.iter_mut().zip(t_in) {
                    *res_ref = res_ref.wrapping_sub(t_in_val.wrapping_mul(n));
                }
            }

            /// Add to res the multiplication of t_in by a scalar n
            /// # Description
            /// res <- res + t_in * n
            /// # Arguments
            /// * `res` - Torus slice (output)
            /// * `t_in` - Torus slice
            /// * `n` - integer (u32 or u64)
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // select a scalar
            /// let scal: Torus = 12;
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // add the scalar multiplication
            /// Tensor::add_scalar_mul(&mut res, &t, scal);
            /// ```
            fn add_scalar_mul(res: &mut [$T], t_in: &[$T], n: $T) {
                debug_assert!(res.len() == t_in.len(), "core_api::math::tensor::add_scalar_mul : res.len() = {} != t_in.len() = {}", res.len(), t_in.len()) ;
                for (res_ref, t_in_val) in res.iter_mut().zip(t_in) {
                    *res_ref = res_ref.wrapping_add(t_in_val.wrapping_mul(n));
                }
            }

            /// Compute the torus-boolean scalar product
            /// # Description
            /// t_torus <- sum_i(t_torus\[i\] * t_bool\[i\])
            /// # Arguments
            /// * `t_torus` - Torus slice
            /// * `t_bool` - Torus slice representing an array of booleans
            /// # Output
            /// * a Torus element containing the desired result
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of a binary torus
            /// let mut tb: Vec<Torus> = vec![0; <Torus as SecretKey>::get_secret_key_length(size, 1)];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // compute a binary multisum
            /// let res: Torus = Tensor::binary_multisum(&t, &tb);
            /// ```
            fn binary_multisum(t_torus: &[$T], t_bool: &[$T]) -> $T {
                debug_assert!(t_bool.len() * <$T as Types>::TORUS_BIT >= t_torus.len(), "core_api::math::tensor::binary_multisum : the key is too short") ;
                let mut res: $T = 0;
                for (i, tor) in t_torus.iter().enumerate() {
                    if SecretKey::get_bit(&t_bool, i) {
                        res = res.wrapping_add(*tor);
                    }
                }
                return res;
            }

            /// Takes as input n vectors of Torus element in t_torus, computes every inner product between the i-th Torus vector and the boolean vector t_bool, and adds that value to the i-th case of t_res
            /// # Description
            /// t_res\[j\] <- t_res\[j\] + sum_i(t_torus\[j\]\[i\] * t_bool\[i\])
            /// # Arguments
            /// * `t_res` - Torus slice (output)
            /// * `t_torus` - Torus slice
            /// * `t_bool` - Torus slice representing a boolean slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of a binary torus
            /// let mut tb: Vec<Torus> = vec![0; <Torus as SecretKey>::get_secret_key_length(size, 1)];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // add several binary multisum
            /// Tensor::add_several_binary_multisum(&mut res, &t, &tb);
            /// ```
            fn add_several_binary_multisum(t_res: &mut [$T], t_torus: &[$T], t_bool: &[$T]) {
                // compute the dimension of each vector of Torus elements
                let dimension: usize = t_torus.len() / t_res.len();

                // loop over the Torus vectors
                for (res, tensor) in izip!(t_res.iter_mut(), t_torus.chunks(dimension)) {
                    // loop over coefficients of a Torus vector
                    for (i, elt) in tensor.iter().enumerate() {
                        if SecretKey::get_bit(&t_bool, i) {
                            *res = res.wrapping_add(*elt);
                        }
                    }
                }
            }


            /// Takes as input a vector of Torus element in t_torus, computes the inner product between the Torus vector and the boolean vector t_bool, and return it
            /// # Description
            /// return sum_i(t_torus\[j\]\[i\] * t_bool\[i\])
            /// # Arguments
            /// * `t_torus` - Torus slice
            /// * `t_bool` - Torus slice representing a boolean slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of a binary torus
            /// let mut tb: Vec<Torus> = vec![0; <Torus as SecretKey>::get_secret_key_length(size, 1)];
            ///
            /// // ... (fill the last tensor)
            ///
            ///
            /// // get the binary multisum
            /// let res: Torus = Tensor::get_binary_multisum(&t, &tb);
            /// ```
            fn get_binary_multisum(t_torus: &[$T], t_bool: &[$T]) -> $T {
                debug_assert!(t_bool.len() * <$T as Types>::TORUS_BIT >= t_torus.len(), "core_api::math::tensor::get_binary_multisum : the key is too short") ;
                let mut res: $T = 0;
                // loop over coefficients of a Torus vector
                for (i, elt) in t_torus.iter().enumerate() {
                    if SecretKey::get_bit(&t_bool, i) {
                        res = res.wrapping_add(*elt);
                    }
                }
                res
            }

            /// Takes as input n vectors of Torus element in t_torus, computes every inner product between the i-th Torus vector and the boolean vector t_bool, and subtracts that value to the i-th case of t_res
            /// # Description
            /// t_res\[j\] <- t_res\[j\] - sum_i(t_torus\[j\]\[i\] * t_bool\[i\])
            /// # Arguments
            /// * `t_res` - Torus slice (output)
            /// * `t_torus` - Torus slice
            /// * `t_bool` - Torus slice representing a boolean slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::crypto::SecretKey;
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of a binary torus
            /// let mut tb: Vec<Torus> = vec![0; <Torus as SecretKey>::get_secret_key_length(size, 1)];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // subtract several binary multisum
            /// Tensor::sub_several_binary_multisum(&mut res, &t, &tb);
            /// ```
            fn sub_several_binary_multisum(t_res: &mut [$T], t_torus: &[$T], t_bool: &[$T]) {
                // compute the dimension of each vector of Torus elements
                let dimension: usize = t_torus.len() / t_res.len();

                // loop over the Torus vectors
                for (res, tensor) in izip!(t_res.iter_mut(), t_torus.chunks(dimension)) {
                    // loop over coefficients of a Torus vector
                    for (i, elt) in tensor.iter().enumerate() {
                        if SecretKey::get_bit(&t_bool, i) {
                            *res = res.wrapping_sub(*elt);
                        }
                    }
                }
            }

            /// Rounds each element of a Torus slice to the max_level * base_log bit
            /// # Warning
            /// * this function panics if TORUS_BIT == max_level * base_log
            /// # Arguments
            /// * `t_res` - Torus slice (output)
            /// * `t_input` - Torus slice
            /// * `base_log` - the number of bit of the basis for the decomposition
            /// * `max_level` - the total number of levels
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            /// let base_log: usize = 2;
            /// let max_level: usize = 4;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // rounding
            /// Tensor::round_to_closest_multiple(&mut res, &t, base_log, max_level);
            /// ```
            fn round_to_closest_multiple(
                t_res: &mut [$T],
                t_input: &[$T],
                base_log: usize,
                max_level: usize,
            ) {
                debug_assert!(t_res.len() == t_input.len(), "core_api::math::tensor::round_to_closest_multiple : t_res.len() = {} =! t_input.len() = {}", t_res.len(), t_input.len()) ;

                // number of bit to throw out
                let shift: usize = <$T as Types>::TORUS_BIT - max_level * base_log;
                let mask: $T = 1 << (shift - 1);
                let mut b: $T;
                for (res, x) in t_res.iter_mut().zip(t_input.iter()) {
                    // get the MSB of the bits that we throw out
                    b = (x & mask) >> (shift - 1);
                    *res = x >> shift;
                    *res += b;
                    *res <<= shift;
                }
            }

            /// Convert a Torus tensor into a float tensor
            /// # Description
            /// t_res <- (float) t_int
            /// # Arguments
            /// * `t_res` - float slice (output)
            /// * `t_int` - Torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of the result
            /// let mut res: Vec<f64> = vec![0.; size];
            ///
            /// // convert
            /// Tensor::int_to_float(&mut res, &t);
            /// ```
            fn int_to_float(t_res: &mut [f64], t_int: &[$T]) {
                debug_assert!(t_res.len() == t_int.len(), "core_api::math::tensor::int_to_float : t_res.len() = {} =! t_int.len() = {}", t_res.len(), t_int.len()) ;
                for (res_ref, int_val) in t_res.iter_mut().zip(t_int) {
                    *res_ref = Types::torus_to_f64(*int_val);
                }
            }

            /// Convert a float tensor into a torus tensor
            /// # Description
            /// t_res <- (Torus) t_float
            /// # Arguments
            /// * `t_res` - Torus slice (output)
            /// * `t_float` - float slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of one tensor
            /// let mut t: Vec<f64> = vec![0.; size];
            ///
            /// // ... (fill the last tensor)
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // convert
            /// Tensor::float_to_int(&mut res, &t);
            /// ```
            #[allow(dead_code)]
            fn float_to_int(t_res: &mut [$T], t_float: &[f64]) {
                debug_assert!(t_res.len() == t_float.len(), "core_api::math::tensor::float_to_int : t_res.len() = {} =! t_float.len() = {}", t_res.len(), t_float.len()) ;
                for (res_ref, float_val) in t_res.iter_mut().zip(t_float) {
                    *res_ref = Types::f64_to_torus(*float_val);
                }
            }

            /// Computes the minimal modular distance element wise between two slices of torus elements
            /// The result is stored in t_res
            /// # Arguments
            /// * `t_res` - torus slice (output)
            /// * `t_x` - torus slice
            /// * `t_y` - torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensors)
            ///
            /// // allocation of the result
            /// let mut res: Vec<Torus> = vec![0; size];
            ///
            /// // convert
            /// Tensor::compute_modular_distance(&mut res, &t0, &t1);
            /// ```
            fn compute_modular_distance(t_res: &mut [$T], t_x: &[$T], t_y: &[$T]) {
                debug_assert!(t_res.len() == t_x.len(), "core_api::math::tensor::compute_modular_distance : t_res.len() = {} =! t_x.len() = {}", t_res.len(), t_x.len()) ;
                debug_assert!(t_res.len() == t_y.len(), "core_api::math::tensor::compute_modular_distance : t_res.len() = {} =! t_y.len() = {}", t_res.len(), t_y.len()) ;
                for (res, (x, y)) in t_res.iter_mut().zip(t_x.iter().zip(t_y.iter())) {
                    let d0 = x.wrapping_sub(*y);
                    let d1 = y.wrapping_sub(*x);
                    *res = d0.min(d1);
                }
            }

            /// Computes a signed modular distance element wise between two slices of torus elements
            /// The result is stored in t_res
            /// # Arguments
            /// * `t_res` - float slice (output)
            /// * `t_x` - torus slice
            /// * `t_y` - torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 100;
            ///
            /// // allocation of two tensors
            /// let mut t0: Vec<Torus> = vec![0; size];
            /// let mut t1: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensors)
            ///
            /// // allocation of the result
            /// let mut res: Vec<f64> = vec![0.; size];
            ///
            /// // convert
            /// Tensor::compute_signed_modular_distance(&mut res, &t0, &t1);
            /// ```
            fn compute_signed_modular_distance(t_res: &mut [f64], t_x: &[$T], t_y: &[$T]) {
                debug_assert!(t_res.len() == t_x.len(), "core_api::math::tensor::compute_signed_modular_distance : t_res.len() = {} =! t_x.len() = {}", t_res.len(), t_x.len()) ;
                debug_assert!(t_res.len() == t_y.len(), "core_api::math::tensor::compute_signed_modular_distance : t_res.len() = {} =! t_y.len() = {}", t_res.len(), t_y.len()) ;
                for (res, (x, y)) in t_res.iter_mut().zip(t_x.iter().zip(t_y.iter())) {
                    let d0 = x.wrapping_sub(*y);
                    let d1 = y.wrapping_sub(*x);
                    let interm = d0.min(d1);
                    if d0 == interm {
                        *res = interm as f64 / f64::powi(2., <$T as Types>::TORUS_BIT as i32);
                    } else {
                        *res = -(interm as f64) / f64::powi(2., <$T as Types>::TORUS_BIT as i32);
                    }
                }
            }

            /// Prints in a pretty way a tensor of Torus elements
            /// # Arguments
            /// * `tensor` - Torus slice
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 4;
            ///
            /// // allocation of a tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensors)
            ///
            /// // print
            /// Tensor::print(&t);
            ///
            /// // stdout:
            /// // [TENSOR] val: 0 0 0 0
            /// ```
            fn print(tensor: &[$T]) {
                print!("[TENSOR] val: ");
                for elt in tensor {
                    print!("{} ", elt);
                }
            }

            /// Writes in a compact way a tensor of Torus elements inside a file
            /// # Arguments
            /// * `tensor` - Torus slice
            /// * `path` - string containing the path of the file we write in
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // setting
            /// let size: usize = 128;
            ///
            /// // create a tensor
            /// let mut tensor: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the tensor)
            ///
            /// // writes in a file
            /// Tensor::write_in_file(&tensor, "test_read_write_torus.txt");
            /// ```
            #[allow(dead_code)]
            fn write_in_file(tensor: &[$T], path: &str) -> std::io::Result<()> {
                let mut file = File::create(path)?;
                let mut bytes: [u8; <$T as Types>::TORUS_BIT / 8];
                for val in tensor.iter() {
                    bytes = unsafe { transmute(val.to_be()) };
                    file.write(&bytes)?;
                }
                Ok(())
            }

            /// Reads in a compact way a tensor of Torus elements inside a file
            /// # Arguments
            /// * `tensor` - Torus slice (output)
            /// * `path` - string containing the path of the file we read in
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // setting
            /// let size: usize = 128;
            ///
            /// // create a tensor
            /// let mut tensor: Vec<Torus> = vec![0; size];
            ///
            /// // reads in a file
            /// Tensor::read_in_file(&mut tensor, "test_read_write_torus.txt");
            /// ```
            #[allow(dead_code)]
            fn read_in_file(tensor: &mut [$T], path: &str) -> std::io::Result<()> {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .open(path)
                    .unwrap();
                let mut bytes: [u8; <$T as Types>::TORUS_BIT / 8] =
                    [0; <$T as Types>::TORUS_BIT / 8];
                for val in tensor.iter_mut() {
                    file.read(&mut bytes)?;
                    bytes.reverse(); // the order is wrong ...
                    *val = unsafe { transmute::<[u8; <$T as Types>::TORUS_BIT / 8], $T>(bytes) };
                }
                Ok(())
            }

            /// Generate a pretty string representing a tensor of Torus elements with their binary representations
            /// base_log defines the size of the bit blocks in the binary representation
            /// # Arguments
            /// * `tensor` - Torus slice
            /// * `base_log` - size of bit blocks in the binary representation
            /// # Example
            /// ```rust
            /// use concrete_lib::core_api::math::Tensor;
            #[doc = $DOC]
            ///
            /// // settings
            /// let size: usize = 4;
            ///
            /// // allocation of a tensor
            /// let mut t: Vec<Torus> = vec![0; size];
            ///
            /// // ... (fill the last tensors)
            ///
            /// // print
            /// let s = Tensor::to_string_torus_binary_representation(&t, 4);
            ///
            /// // s = "0000 0000 0000 0000 0000 0000 0000 0000:0000 0000 0000 0000 0000 0000 0000 0000:0000 0000 0000 0000 0000 0000 0000 0000:0000 0000 0000 0000 0000 0000 0000 0000:"
            /// ```
            fn to_string_torus_binary_representation(tensor: &[$T], base_log: usize) -> String {
                let mut res = String::from("");
                for elt in tensor.iter() {
                    res = [res, Types::torus_binary_representation(*elt, base_log)].concat();
                    res.push(':');
                }
                return res;
            }
        }
    };
}

impl_trait_tensor!(u32, "type Torus = u32;");
impl_trait_tensor!(u64, "type Torus = u64;");

/// Writes in a compact way a tensor of FTorus elements inside a file
/// # Arguments
/// * `tensor` - FTorus slice
/// * `path` - string containing the path of the file we write in
/// # Example
/// ```rust
/// use concrete_lib::core_api::math::tensor;
/// use concrete_lib::types::FTorus;
///
/// // setting
/// let size: usize = 128;
///
/// // create a tensor
/// let mut tensor: Vec<FTorus> = vec![0.; size];
///
/// // ... (fill the tensor)
///
/// // writes in a file
/// tensor::write_in_file_ftorus(&tensor, "test_read_write_ftorus.txt");
/// ```
#[allow(dead_code)]
pub fn write_in_file_ftorus(tensor: &[FTorus], path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let mut bytes: [u8; FTORUS_BIT / 8];
    for val in tensor.iter() {
        bytes = unsafe { transmute(val.to_bits()) };
        file.write(&bytes)?;
    }
    Ok(())
}

/// Writes in a compact way a tensor of CTorus elements inside a file
/// # Arguments
/// * `tensor` - CTorus slice
/// * `path` - string containing the path of the file we write in
/// # Example
/// ```rust
/// use concrete_lib::core_api::math::tensor;
/// use concrete_lib::types::CTorus;
/// use num_traits::Zero;
///
/// // setting
/// let size: usize = 128;
///
/// // create a tensor
/// let mut tensor: Vec<CTorus> = vec![CTorus::zero(); size];
///
/// // ... (fill the tensor)
///
/// // writes in a file
/// tensor::write_in_file_ctorus(&tensor, "test_read_write_ctorus.txt");
/// ```
#[allow(dead_code)]
pub fn write_in_file_ctorus(tensor: &[CTorus], path: &str) -> std::io::Result<()> {
    let mut f_tensor: Vec<FTorus> = vec![0.; tensor.len() * 2];
    for (couple, c) in f_tensor.chunks_mut(2).zip(tensor.iter()) {
        couple[0] = c.re;
        couple[1] = c.im;
    }
    write_in_file_ftorus(&f_tensor, path).unwrap();
    Ok(())
}

/// Reads in a compact way a tensor of FTorus elements inside a file
/// # Arguments
/// * `tensor` - FTorus slice (output)
/// * `path` - string containing the path of the file we read in
/// # Example
/// ```rust
/// use concrete_lib::core_api::math::tensor;
/// use concrete_lib::types::FTorus;
///
/// // setting
/// let size: usize = 128;
///
/// // create a tensor
/// let mut tensor: Vec<FTorus> = vec![0.; size];
///
/// // reads in a file
/// tensor::read_in_file_ftorus(&mut tensor, "test_read_write_ftorus.txt");
/// ```
#[allow(dead_code)]
pub fn read_in_file_ftorus(tensor: &mut [FTorus], path: &str) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path)
        .unwrap();
    let mut bytes: [u8; FTORUS_BIT / 8] = [0; FTORUS_BIT / 8];
    for val in tensor.iter_mut() {
        file.read(&mut bytes)?;
        *val = unsafe { transmute::<[u8; FTORUS_BIT / 8], FTorus>(bytes) };
    }
    Ok(())
}

/// Reads in a compact way a tensor of CTorus elements inside a file
/// # Arguments
/// * `tensor` - CTorus slice (output)
/// * `path` - string containing the path of the file we read in
/// # Example
/// ```rust
/// use concrete_lib::core_api::math::tensor;
/// use concrete_lib::types::CTorus;
/// use num_traits::Zero;
///
/// // setting
/// let size: usize = 128;
///
/// // create a tensor
/// let mut tensor: Vec<CTorus> = vec![CTorus::zero(); size];
///
/// // reads in a file
/// tensor::read_in_file_ctorus(&mut tensor, "test_read_write_ftorus.txt");
/// ```
#[allow(dead_code)]
pub fn read_in_file_ctorus(tensor: &mut [CTorus], path: &str) -> std::io::Result<()> {
    let mut f_tensor: Vec<FTorus> = vec![0.; tensor.len() * 2];
    read_in_file_ftorus(&mut f_tensor, path).unwrap();
    for (couple, c) in f_tensor.chunks(2).zip(tensor.iter_mut()) {
        *c = CTorus::new(couple[0], couple[1]);
    }
    Ok(())
}
