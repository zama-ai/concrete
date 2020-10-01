//! Random Tensor Operations
//! * Contains every function related to random tensors

#[cfg(test)]
mod tests;

use crate::Types;
use num_integer::Integer;
use openssl;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::slice;
pub trait Random: Sized {
    fn rng_uniform_n_msb(t: &mut [Self], n: usize);
    fn rng_uniform_n_lsb(k: &mut [Self], n: usize);
    fn rng_uniform(k: &mut [Self]);
    fn openssl_uniform(k: &mut [Self]);
    fn rng_uniform_with_some_zeros(k: &mut [Self], probability: Self);
    fn vectorial_rng_normal(k: &mut [Self], mean: f64, std_dev: f64);
    fn vectorial_openssl_normal(k: &mut [Self], mean: f64, std_dev: f64);
    fn rng_normal(mean: f64, std_dev: f64) -> Self;
    fn openssl_normal(mean: f64, std_dev: f64) -> Self;
    fn openssl_normal_couple(mean: f64, std_dev: f64) -> (Self, Self);
}

macro_rules! impl_trait_random {
    ($T:ty,$DOC:expr) => {
        impl Random for $T {
            /// Fills a Torus tensor with uniform random in [0,2**n[ on the MSB
            /// # Arguments
            /// * `t` - a Torus tensor (output)
            /// * `n` - number of bit to fill with random
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // allocation of a torus
            /// let mut t: Vec<Torus> = vec![0; 100];
            ///
            /// // fill with uniform random in the MSB (5 bits)
            /// Random::rng_uniform_n_msb(&mut t, 5);
            /// ```
            fn rng_uniform_n_msb(t: &mut [$T], n: usize) {
                let mut rng = rand::thread_rng();
                for ti in t.iter_mut() {
                    *ti = rng.gen::<$T>() << (<$T as Types>::TORUS_BIT - n);
                }
            }

            /// Fills a Torus tensor with uniform random in [0,2**n[ on the LSB
            /// # Arguments
            /// * `k` - a Torus tensor (output)
            /// * `n` - number of bit to fill with random
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // allocation of a torus
            /// let mut t: Vec<Torus> = vec![0; 100];
            ///
            /// // fill with uniform random in the LSB (5 bits)
            /// Random::rng_uniform_n_lsb(&mut t, 5);
            /// ```
            fn rng_uniform_n_lsb(k: &mut [$T], n: usize) {
                let mut rng = rand::thread_rng();
                for ki_ref in k.iter_mut() {
                    *ki_ref = rng.gen::<$T>() >> (<$T as Types>::TORUS_BIT - n);
                }
            }

            /// Fills a Torus tensor with uniform random
            /// # Arguments
            /// * `k` - a Torus tensor (output)
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // allocation of a torus
            /// let mut t: Vec<Torus> = vec![0; 100];
            ///
            /// // fill with uniform random
            /// Random::rng_uniform(&mut t);
            /// ```
            fn rng_uniform(k: &mut [$T]) {
                let mut rng = rand::thread_rng();
                for ki_ref in k.iter_mut() {
                    *ki_ref = rng.gen::<$T>();
                }
            }

            /// Fills a Torus tensor with uniform random using openssl CSPTRNG
            /// # Arguments
            /// * `k` - a Torus tensor (output)
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // allocation of a torus
            /// let mut t: Vec<Torus> = vec![0; 100];
            ///
            /// // fill with uniform random
            /// Random::openssl_uniform(&mut t);
            /// ```
            fn openssl_uniform(k: &mut [$T]) {
                let n: usize = k.len( ) ;
                let nb_bytes = <$T as Types>::TORUS_BIT / 8 ;
                let k_bytes = unsafe { slice::from_raw_parts_mut(k.as_mut_ptr() as *mut u8, n * nb_bytes)};
                openssl::rand::rand_bytes(k_bytes).unwrap() ;
            }

            /// Either fills elements of k with uniform random or with a zero according to probability
            /// if probability is set to 0.2 (0.2 * 2**TORUS_BIT), then there will be approximately 1/5 random elements
            /// and the rest will be set to zero
            /// # Arguments
            /// * `k` - Torus slice (output)
            /// * `probability` - the probability for an element to be uniformly random
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            /// use concrete_lib::Types;
            ///
            /// // allocation of a torus
            /// let mut t: Vec<Torus> = vec![0; 100];
            ///
            /// // fill with uniform random 20 percents of the time
            /// Random::rng_uniform_with_some_zeros(
            ///     &mut t,
            ///     (0.2 * f64::powi(2., <Torus as Types>::TORUS_BIT as i32)) as Torus,
            /// );
            /// ```
            fn rng_uniform_with_some_zeros(k: &mut [$T], probability: $T) {
                let mut rng = rand::thread_rng();
                for ki_ref in k.iter_mut() {
                    *ki_ref = rng.gen::<$T>();
                    if *ki_ref <= probability {
                        *ki_ref = rng.gen::<$T>();
                    } else {
                        *ki_ref = 0;
                    }
                }
            }
            /// Fills a Torus tensor with normal random
            /// # Arguments
            /// * `k` - a Torus tensor (output)
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std_dev: f64 = f64::powi(2., -20);
            /// let k = 100;
            ///
            /// // allocation of a tensor
            /// let mut t: Vec<Torus> = vec![0; k];
            ///
            /// // fill with random
            /// Random::vectorial_rng_normal(&mut t, 0., std_dev);
            /// ```
            fn vectorial_rng_normal(k: &mut [$T], mean: f64, std_dev: f64) {
                let normal = Normal::new(mean, std_dev).unwrap();
                for k_ref in k.iter_mut() {
                    *k_ref = Types::f64_to_torus(normal.sample(&mut rand::thread_rng()));
                }
            }

            /// Fills a Torus tensor with normal random using a CSPRNG and Box-Muller algorithm
            /// # Arguments
            /// * `k` - a Torus tensor (output)
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std_dev: f64 = f64::powi(2., -20);
            /// let k = 100;
            ///
            /// // allocation of a tensor
            /// let mut t: Vec<Torus> = vec![0; k];
            ///
            /// // fill with random
            /// Random::vectorial_openssl_normal(&mut t, 0., std_dev);
            /// ```
            fn vectorial_openssl_normal(k: &mut [$T], mean: f64, std_dev: f64) {

                let is_even: bool = k.len().is_even() ;

                for k_ref in k.chunks_exact_mut(2) {
                    let (u,v) = Self::openssl_normal_couple(mean, std_dev) ;
                    k_ref[0] = u ;
                    k_ref[1] = v;
                }
                if !is_even {
                    let (u,_v) = Self::openssl_normal_couple(mean, std_dev) ;
                    k[k.len() - 1] = u ;
                }


            }

            /// Returns a Torus element with normal random
            /// # Arguments
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Output
            /// * returns the Torus element sampled from the desired distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std_dev: f64 = f64::powi(2., -20);
            /// let k = 100;
            ///
            /// // get a random element
            /// let elt: Torus = Random::rng_normal(0., std_dev);
            /// ```
            fn rng_normal(mean: f64, std_dev: f64) -> $T {
                let normal = Normal::new(mean, std_dev).unwrap();
                return Types::f64_to_torus(normal.sample(&mut rand::thread_rng()));
            }

            /// Returns a Torus element with normal random using a CSPRNG and Box-Muller algorithm
            /// # Warning
            /// * This function discards 1/2 of the sample
            /// # Arguments
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Output
            /// * returns the Torus element sampled from the desired distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std_dev: f64 = f64::powi(2., -20);
            /// let k = 100;
            ///
            /// // get a random element
            /// let elt: Torus = Random::openssl_normal(0., std_dev);
            /// ```
            fn openssl_normal(mean: f64, std_dev: f64) -> $T {
                let mut uniform_rand = vec![0i64; 2] ;
                let mut res = 0 ;
                let mut finish: bool = false ;
                while !finish {
                    {
                        let uniform_rand_bytes = unsafe { slice::from_raw_parts_mut(uniform_rand.as_mut_ptr() as *mut u8, 2 * 8)};
                        openssl::rand::rand_bytes(uniform_rand_bytes).unwrap() ;
                    }
                    let u: f64 = (uniform_rand[0] as f64) * f64::powi(2., -63) ;
                    let v: f64 = (uniform_rand[1] as f64) * f64::powi(2., -63) ;
                    let s: f64 = f64::powi(u, 2) + f64::powi(v, 2) ;
                    if (s>0. && s<1.) {
                        let cst: f64 = std_dev * f64::sqrt(-2. * f64::ln(s) / s);
                        res = Types::f64_to_torus(u * cst + mean) ;
                        finish = true ;
                    }
                }
                return res ;

            }

            /// Returns a Torus element with normal random using a CSPRNG and Box-Muller algorithm
            /// # Arguments
            /// * `mean` - mean of the normal distribution
            /// * `std_dev` - standard deviation of the normal distribution
            /// # Output
            /// * returns the Torus element sampled from the desired distribution
            /// # Example
            /// ```rust
            /// use concrete_lib::operators::math::Random;
            #[doc = $DOC]
            ///
            /// // settings
            /// let std_dev: f64 = f64::powi(2., -20);
            /// let k = 100;
            ///
            /// // get a random element
            /// let elt: Torus = Random::openssl_normal(0., std_dev);
            /// ```
            fn openssl_normal_couple(mean: f64, std_dev: f64) -> ($T, $T) {
                let mut uniform_rand = vec![0i64; 2] ;
                let mut res_0 = 0 ;
                let mut res_1 = 0 ;
                let mut finish: bool = false ;
                while !finish {
                    {
                        let uniform_rand_bytes = unsafe { slice::from_raw_parts_mut(uniform_rand.as_mut_ptr() as *mut u8, 2 * 8)};
                        openssl::rand::rand_bytes(uniform_rand_bytes).unwrap() ;
                    }
                    let u: f64 = (uniform_rand[0] as f64) * f64::powi(2., -63) ;
                    let v: f64 = (uniform_rand[1] as f64) * f64::powi(2., -63) ;
                    let s: f64 = f64::powi(u, 2) + f64::powi(v, 2) ;
                    if (s>0. && s<1.) {
                        let cst: f64 = std_dev * f64::sqrt(-2. * f64::ln(s) / s);
                        res_0 = Types::f64_to_torus(u * cst + mean) ;
                        res_1 = Types::f64_to_torus(v * cst + mean) ;
                        finish = true ;
                    }
                }
                return (res_0, res_1) ;

            }

        }
    };
}

/// Returns a Torus element with normal random using a CSPRNG and Box-Muller algorithm
/// # Arguments
/// * `mean` - mean of the normal distribution
/// * `std_dev` - standard deviation of the normal distribution
/// # Output
/// * returns the Torus element sampled from the desired distribution
/// # Example
/// ```rust
/// use concrete_lib::operators::math::random;
///
/// // settings
/// let std_dev: f64 = f64::powi(2., -20);
/// let mean = 0.;
///
/// // draw random sample
/// let (r1, r2): (f64, f64) = random::openssl_float_normal_couple(mean, std_dev);
/// ```
pub fn openssl_float_normal_couple(mean: f64, std_dev: f64) -> (f64, f64) {
    let mut uniform_rand = vec![0i64; 2];
    let mut res_0 = 0.;
    let mut res_1 = 0.;
    let mut finish: bool = false;
    while !finish {
        {
            let uniform_rand_bytes =
                unsafe { slice::from_raw_parts_mut(uniform_rand.as_mut_ptr() as *mut u8, 2 * 8) };
            openssl::rand::rand_bytes(uniform_rand_bytes).unwrap();
        }
        let u: f64 = (uniform_rand[0] as f64) * f64::powi(2., -63);
        let v: f64 = (uniform_rand[1] as f64) * f64::powi(2., -63);
        let s: f64 = f64::powi(u, 2) + f64::powi(v, 2);
        if s > 0. && s < 1. {
            let cst: f64 = std_dev * f64::sqrt(-2. * f64::ln(s) / s);
            res_0 = u * cst + mean;
            res_1 = v * cst + mean;
            finish = true;
        }
    }
    return (res_0, res_1);
}
/// Fills a float tensor with normal random
/// # Arguments
/// * `k` - a float tensor (output)
/// * `mean` - mean of the normal distribution
/// * `std_dev` - standard deviation of the normal distribution
/// # Example
/// ```rust
/// use concrete_lib::operators::math::random;
///
/// // settings
/// let std_dev: f64 = f64::powi(2., -20);
/// let k = 100;
///
/// // allocation of a tensor
/// let mut t: Vec<f64> = vec![0.; k];
///
/// // fill with random
/// random::vectorial_rng_float_normal(&mut t, 0., std_dev);
/// ```
pub fn vectorial_rng_float_normal(k: &mut [f64], mean: f64, std_dev: f64) {
    let normal = Normal::new(mean, std_dev).unwrap();

    for k_ref in k.iter_mut() {
        *k_ref = normal.sample(&mut rand::thread_rng());
    }
}

/// Return a sample drawn from a normal distribution
/// # Arguments
/// * `mean` - mean of the normal distribution
/// * `std_dev` - standard deviation of the normal distribution
/// # Output
/// * Return the sample
/// # Example
/// ```rust
/// use concrete_lib::operators::math::random;
///
/// // settings
/// let std_dev: f64 = f64::powi(2., -20);
/// let mean = 0.;
///
/// let r: f64 = random::rng_float_normal(mean, std_dev);
/// ```
pub fn rng_float_normal(mean: f64, std_dev: f64) -> f64 {
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.sample(&mut rand::thread_rng())
}

/// Fills a float tensor with normal random
/// # Arguments
/// * `k` - a float tensor (output)
/// * `mean` - mean of the normal distribution
/// * `std_dev` - standard deviation of the normal distribution
/// # Example
/// ```rust
/// use concrete_lib::operators::math::random;
///
/// // settings
/// let std_dev: f64 = f64::powi(2., -20);
/// let k = 100;
///
/// // allocation of a tensor
/// let mut t: Vec<f64> = vec![0.; k];
///
/// // fill with random
/// random::vectorial_openssl_float_normal(&mut t, 0., std_dev);
/// ```
pub fn vectorial_openssl_float_normal(k: &mut [f64], mean: f64, std_dev: f64) {
    let is_even: bool = k.len().is_even();

    for k_ref in k.chunks_exact_mut(2) {
        let (u, v) = openssl_float_normal_couple(mean, std_dev);
        k_ref[0] = u;
        k_ref[1] = v;
    }
    if !is_even {
        let (u, _v) = openssl_float_normal_couple(mean, std_dev);
        k[k.len() - 1] = u;
    }
}

impl_trait_random!(u32, "type Torus = u32;");
impl_trait_random!(u64, "type Torus = u64;");
