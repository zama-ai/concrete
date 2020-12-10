//! Noise formulas for the lwe-sample related operations
//! Those functions will be used in the lwe tests to check that
//! the noise behavior is consistent with the theory.

use itertools::izip;

pub trait LWE: Sized {
    type STorus;
    fn single_scalar_mul(variance: f64, n: Self) -> f64;
    fn scalar_mul(var_out: &mut [f64], var_in: &[f64], t: &[Self]);
    fn scalar_mul_inplace(var: &mut [f64], t: &[Self]);
    fn multisum_uncorrelated(variances: &[f64], weights: &[Self]) -> f64;
    fn key_switch(
        dimension_before: usize,
        l_ks: usize,
        base_log: usize,
        var_ks: f64,
        var_input_lwe: f64,
    ) -> f64;
}

macro_rules! impl_trait_npe_lwe {
    ($T:ty,$S:ty,$DOC:expr) => {
        impl LWE for $T {
            type STorus = $S;
            /// Return the variance of the keyswitch on a LWE sample given a set of parameters.
            /// To see how to use it, please refer to the test of the keyswitch
            /// # Warning
            /// * This function compute the noise of the keyswitch without functional evaluation
            /// # Arguments
            /// `dimension_before` - size of the input LWE mask
            /// `l_ks` - number of level max for the torus decomposition
            /// `base_log` - number of bits for the base B (B=2^base_log)
            /// `var_ks` - variance of the keyswitching key
            /// `var_input` - variance of the input LWE
            /// # Example
            /// ```rust
            /// use concrete_npe::LWE ;
            #[doc = $DOC]
            /// // settings
            /// let dimension_before: usize = 630 ;
            /// let l_ks: usize = 4 ;
            /// let base_log: usize = 7 ;
            /// let var_ks: f64 = f64::powi(2., -38) ;
            /// let var_input: f64 = f64::powi(2., -40) ;
            /// // Computing the noise
            /// let var_ks = <Torus as LWE>::key_switch(dimension_before, l_ks, base_log, var_ks, var_input) ;
            /// ```
            fn key_switch(
                dimension_before: usize,
                l_ks: usize,
                base_log: usize,
                var_ks: f64,
                var_input: f64,
            ) -> f64 {
                let q_square = f64::powi(2., (2 * std::mem::size_of::<$T>()) as i32);
                let res_1: f64 = dimension_before as f64
                    * (1. / 24. * f64::powi(2.0, -2 * (base_log * l_ks) as i32)
                        + 1. / (48. * q_square));
                let res_2: f64 = dimension_before as f64
                    * l_ks as f64
                    * (f64::powi(2., 2 * base_log as i32) / 12. + 1. / 6.)
                    * var_ks;

                let res: f64 = var_input + res_1 + res_2;
                return res;
            }

            /// Computes the variance of the error distribution after a multiplication of a ciphertext by a scalar
            /// i.e. sigma_out^2 <- n^2 * sigma^2
            /// Arguments
            /// * `variance` - variance of the input LWE
            /// * `n` - a signed integer
            /// Output
            /// * the output variance
            /// # Example
            /// ```rust
            /// use concrete_npe::LWE ;
            #[doc = $DOC]
            /// // parameters
            /// let variance: f64 = f64::powi(2., -48) ;
            /// let n: Torus = (-543 as i64) as Torus ;
            /// // noise computation
            /// let noise: f64 = <Torus as LWE>::single_scalar_mul(variance, n) ;
            /// ```
            fn single_scalar_mul(variance: f64, n: Self) -> f64 {
                let sn: Self::STorus = n as Self::STorus;
                return variance * ((sn * sn) as f64);
            }

            /// Computes the variance of the error distribution after a multisum between uncorrelated ciphertexts and scalar weights
            /// i.e. sigma_out^2 <- \Sum_i weight_i^2 * sigma_i^2
            /// Arguments
            /// * `variances` - a slice of f64 with the error variances of all the input uncorrelated ciphertexts
            /// * `weights` - a slice of Torus with the input weights
            /// Output
            /// * the output variance
            /// # Example
            /// ```rust
            /// use concrete_npe::LWE ;
            #[doc = $DOC]
            /// // parameters
            /// let variances: Vec<f64> = vec![f64::powi(2., -30), f64::powi(2., -32)] ;
            /// let weights: Vec<Torus> = vec![(-543 as i64) as Torus, 10 as Torus] ;
            /// // noise computation
            /// let noise: f64 = <Torus as LWE>::multisum_uncorrelated(&variances, &weights) ;
            /// ```
            fn multisum_uncorrelated(variances: &[f64], weights: &[Self]) -> f64 {
                let mut new_variance: f64 = 0.;

                for (var_ref, &w) in variances.iter().zip(weights) {
                    new_variance += Self::single_scalar_mul(*var_ref, w);
                }

                return new_variance;
            }

            /// Computes the variance of the error distribution after a multiplication of several ciphertexts by several scalars
            /// Arguments
            /// * `var_out` - variances of the output LWEs (output)
            /// * `var_in` - variances of the input LWEs
            /// * `t` - a slice of signed integer
            /// # Example
            /// ```rust
            /// use concrete_npe::LWE ;
            #[doc = $DOC]
            /// // parameters
            /// let var_in: Vec<f64> = vec![f64::powi(2., -30), f64::powi(2., -32)] ;
            /// let weights: Vec<Torus> = vec![(-543 as i64) as Torus, 10 as Torus] ;
            /// // noise computation
            /// let mut var_out: Vec<f64> = vec![0.; 2] ;
            /// <Torus as LWE>::scalar_mul(&mut var_out, &var_in, &weights) ;
            /// ```
            fn scalar_mul(var_out: &mut [f64], var_in: &[f64], t: &[Self]) {
                for (vo, vi, tval) in izip!(var_out.iter_mut(), var_in.iter(), t.iter()) {
                    *vo = Self::single_scalar_mul(*vi, *tval);
                }
            }

            /// Computes the variance of the error distribution after a multiplication of several ciphertexts by several scalars
            /// Arguments
            /// * `var` - variances of the input/output LWEs (output)
            /// * `t` - a slice of signed integer
            /// # Example
            /// ```rust
            /// use concrete_npe::LWE ;
            #[doc = $DOC]
            /// // parameters
            /// let mut var: Vec<f64> = vec![f64::powi(2., -30), f64::powi(2., -32)] ;
            /// let weights: Vec<Torus> = vec![(-543 as i64) as Torus, 10 as Torus] ;
            /// // noise computation
            /// <Torus as LWE>::scalar_mul_inplace(&mut var, &weights) ;
            /// ```
            fn scalar_mul_inplace(var: &mut [f64], t: &[Self]) {
                for (v, tval) in izip!(var.iter_mut(), t.iter()) {
                    *v = Self::single_scalar_mul(*v, *tval);
                }
            }
        }
    };
}

impl_trait_npe_lwe!(u32, i32, "type Torus = u32;");
impl_trait_npe_lwe!(u64, i64, "type Torus = u64;");

/// Computes the variance of the error distribution after an addition of two uncorrelated ciphertexts
/// sigma_out^2 <- sigma0^2 + sigma1^2
/// Arguments
/// * `variance_0` - variance of the error of the first input ciphertext
/// * `variance_1` - variance of the error of the second input ciphertext
/// Output
/// * the sum of the variances
pub fn add_2_uncorrelated(variance_0: f64, variance_1: f64) -> f64 {
    variance_0 + variance_1
}

/// Computes the variance of the error distribution after an addition of n uncorrelated ciphertexts
/// sigma_out^2 <- \Sum sigma_i^2
/// Arguments
/// * `variances` - a slice of f64 with the error variances of all the input uncorrelated ciphertexts
/// Output
/// * the sum of the variances
pub fn add_n_uncorrelated(variances: &[f64]) -> f64 {
    let mut new_variance: f64 = 0.;
    for var in variances.iter() {
        new_variance += *var;
    }
    new_variance
}

/// Computes an upper bound for the number of 1 in a secret key
/// z*sigma + mean
pub fn upper_bound_hw_secret_key(n: usize) -> usize {
    let n_f: f64 = n as f64;
    let sigma: f64 = f64::sqrt(n_f) / 2.;
    let mean: f64 = n_f / 2.;
    let z: f64 = 3.;
    (mean + z * sigma).round() as usize
}

/// Computes an upper bound for the log2 of the rounding noise
/// z*sigma (mean is 0.)
/// 2048 -> 4.838859820820504
/// 1024 -> 4.357122758833062
///  630 -> 4.024243436996168
///  512 -> 3.8824357953680453
pub fn log2_rounding_noise(n: usize) -> f64 {
    let bound_sup: f64 = upper_bound_hw_secret_key(n) as f64;
    let sigma: f64 = f64::sqrt(bound_sup / 12.);
    let z: f64 = 3.;
    f64::log2(sigma * z)
}
