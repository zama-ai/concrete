//! Low-overhead fhe library.
//!
//! Welcome to the `concrete-core` documentation!
//!
//! # Fully Homomorphic Encryption
//!
//! This library contains low-level primitives which can be used to implement *fully
//! homomorphically encrypted* programs. In a nutshell, fully homomorphic
//! encryption allows you to perform any computation you would normally perform over clear data;
//! but this time over encrypted data. With fhe, you can perform computations without putting
//! your trust on third-party providers. To learn more about the fhe schemes used in this library,
//! you can have a look at the following papers:
//!
//! + [CONCRETE: Concrete Operates oN Ciphertexts Rapidly by Extending TfhE](https://whitepaper.zama.ai/concrete/WAHC2020Demo.pdf)
//! + [Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks](https://whitepaper.zama.ai/)
//!
//! If you are not accustomed to cryptography, but are still interested by performing you should
//! check the [`concrete`](https://crates.io/crates/concrete) library, which provides a simpler,
//! higher-level API.
//!
//! # Quick Example
//!
//! Despite being low-overhead, `concrete-core` offers a pretty straightforward interface:
//! ```
//! // This examples shows how to multiply a secret value by a public one homomorphically. First
//! // we import the proper symbols:
//! use concrete_core::crypto::encoding::{RealEncoder, Cleartext, Encoder, Plaintext};
//! use concrete_core::crypto::secret::LweSecretKey;
//! use concrete_core::crypto::LweDimension;
//! use concrete_core::crypto::lwe::LweCiphertext;
//! use concrete_core::math::dispersion::LogStandardDev;
//!
//! // We initialize an encoder that will allow us to turn cleartext values into plaintexts.
//! let encoder = RealEncoder{offset: 0., delta: 100.};
//! // Our secret value will be 10.,
//! let cleartext = Cleartext(10_f64);
//! let public_multiplier = Cleartext(5);
//! // We encode our cleartext
//! let plaintext = encoder.encode(cleartext);
//!
//! // We generate a new secret key which is used to encrypt the message
//! let secret_key_size = LweDimension(710);
//! let secret_key = LweSecretKey::generate(secret_key_size);
//!
//! // We allocate a ciphertext and encrypt the plaintext with a secure parameter
//! let mut ciphertext = LweCiphertext::allocate(0u32, secret_key_size.to_lwe_size());
//! secret_key.encrypt_lwe(
//!     &mut ciphertext,
//!     &plaintext,
//!     LogStandardDev::from_log_standard_dev(-17.)
//! );
//!
//! // We perform the homomorphic operation:
//! ciphertext.update_with_scalar_mul(public_multiplier);
//!
//! // We decrypt the message
//! let mut output_plaintext = Plaintext(0u32);
//! secret_key.decrypt_lwe(&mut output_plaintext, &ciphertext);
//! let output_cleartext = encoder.decode(output_plaintext);
//!
//! // We check that the result is as expected !
//! assert!((output_cleartext.0 - 50.).abs() < 0.01);
//! ```
//!
//! The scalar multiplication is only one of the many operations available. For more informations
//! about the operations available, check the [`crypto`] module.

#[allow(unused_macros)]
macro_rules! assert_delta {
    ($A:expr, $B:expr, $d:expr) => {
        for (x, y) in $A.iter().zip($B) {
            if (*x as i64 - y as i64).abs() > $d {
                panic!("{} != {} ", *x, y);
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! assert_delta_scalar {
    ($A:expr, $B:expr, $d:expr) => {
        if ($A as i64 - $B as i64).abs() > $d {
            panic!("{} != {} +- {}", $A, $B, $d);
        }
    };
}

#[allow(unused_macros)]
macro_rules! assert_delta_scalar_float {
    ($A:expr, $B:expr, $d:expr) => {
        if ($A - $B).abs() > $d {
            panic!("{} != {} +- {}", $A, $B, $d);
        }
    };
}

#[allow(unused_macros)]
macro_rules! modular_distance {
    ($A:expr, $B:expr) => {
        ($A.wrapping_sub($B)).min($B.wrapping_sub($A))
    };
}

pub mod crypto;
pub mod math;
pub mod numeric;
pub mod utils;

#[doc(hidden)]
#[cfg(test)]
pub mod test_tools {
    use crate::crypto::{
        CiphertextCount, GlweDimension, LweDimension, PlaintextCount, UnsignedTorus,
    };
    use crate::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
    use crate::math::dispersion::DispersionParameter;
    use crate::math::polynomial::PolynomialSize;
    use crate::math::random;
    use crate::math::random::random_uniform;
    use crate::math::tensor::{AsRefSlice, AsRefTensor};
    use crate::numeric::UnsignedInteger;
    use std::ops::Range;

    fn modular_distance<T: UnsignedInteger>(first: T, other: T) -> T {
        let d0 = first.wrapping_sub(other);
        let d1 = other.wrapping_sub(first);
        std::cmp::min(d0, d1)
    }

    fn torus_modular_distance<T: UnsignedInteger>(first: T, other: T) -> f64 {
        let d0 = first.wrapping_sub(other);
        let d1 = other.wrapping_sub(first);
        if d0 < d1 {
            let d: f64 = d0.cast_into();
            d / 2_f64.powi(T::BITS as i32)
        } else {
            let d: f64 = d1.cast_into();
            -d / 2_f64.powi(T::BITS as i32)
        }
    }

    pub fn assert_delta_std_dev<First, Second, Element>(
        first: &First,
        second: &Second,
        dist: impl DispersionParameter,
    ) where
        First: AsRefTensor<Element = Element>,
        Second: AsRefTensor<Element = Element>,
        Element: UnsignedTorus,
    {
        for (x, y) in first.as_tensor().iter().zip(second.as_tensor().iter()) {
            println!("{:?}, {:?}", *x, *y);
            println!("{}", dist.get_standard_dev());
            let distance: f64 = modular_distance(*x, *y).cast_into();
            let torus_distance = distance / 2_f64.powi(Element::BITS as i32);
            if torus_distance > 5. * dist.get_standard_dev() {
                panic!("{} != {} ", x, y);
            }
        }
    }

    pub fn assert_noise_distribution<First, Second, Element>(
        first: &First,
        second: &Second,
        dist: impl DispersionParameter,
    ) where
        First: AsRefTensor<Element = Element>,
        Second: AsRefTensor<Element = Element>,
        Element: UnsignedTorus,
    {
        use crate::math::tensor::Tensor;

        let std_dev = dist.get_standard_dev();
        let confidence = 0.95;
        let n_slots = first.as_tensor().len();

        // allocate 2 slices: one for the error samples obtained, the second for fresh samples according to the std_dev computed
        let mut sdk_samples = Tensor::allocate(0. as f64, n_slots);

        // recover the errors from each ciphertexts
        sdk_samples.fill_with_two(&first.as_tensor(), &second.as_tensor(), |a, b| {
            torus_modular_distance(*a, *b)
        });

        // fill the theoretical sample vector according to std_dev
        let theoretical_samples = random::random_gaussian_tensor(n_slots, 0., std_dev);

        // compute the kolmogorov smirnov test
        let result = kolmogorov_smirnov::test_f64(
            sdk_samples.as_slice(),
            theoretical_samples.as_slice(),
            confidence,
        );

        if result.is_rejected {
            // compute the mean of our errors
            let mut mean: f64 = sdk_samples.iter().sum();
            mean /= sdk_samples.len() as f64;

            // compute the variance of the errors
            let mut sdk_variance: f64 = sdk_samples.iter().map(|x| f64::powi(x - mean, 2)).sum();
            sdk_variance /= (sdk_samples.len() - 1) as f64;

            // compute the standard deviation
            let sdk_std_log2 = f64::log2(f64::sqrt(sdk_variance)).round();
            let th_std_log2 = f64::log2(std_dev).round();

            // test if theoretical_std_dev > sdk_std_dev
            if sdk_std_log2 > th_std_log2 {
                panic!(
                    "Statistical test failed :
                    -> inputs are not from the same distribution with a probability {}
                    -> sdk_std = {} ; th_std {}.",
                    result.reject_probability, sdk_std_log2, th_std_log2
                );
            }
        }
    }

    /// Returns a random plaintext count.
    pub fn random_plaintext_count(range: Range<usize>) -> PlaintextCount {
        PlaintextCount(random_usize_between(range))
    }

    /// Returns a random ciphertext count.
    pub fn random_ciphertext_count(range: Range<usize>) -> CiphertextCount {
        CiphertextCount(random_usize_between(range))
    }

    /// Returns a random LWE dimension.
    pub fn random_lwe_dimension(range: Range<usize>) -> LweDimension {
        LweDimension(random_usize_between(range))
    }

    /// Returns a random GLWE dimension.
    pub fn random_glwe_dimension(range: Range<usize>) -> GlweDimension {
        GlweDimension(random_usize_between(range))
    }

    /// Returns a random polynomial size.
    pub fn random_polynomial_size(range: Range<usize>) -> PolynomialSize {
        assert!(range.start >= 2);
        PolynomialSize(random_usize_between(range))
    }

    /// Returns a random base log.
    pub fn random_base_log(range: Range<usize>) -> DecompositionBaseLog {
        assert!(range.start >= 2);
        DecompositionBaseLog(random_usize_between(range))
    }

    /// Returns a random level count.
    pub fn random_level_count(range: Range<usize>) -> DecompositionLevelCount {
        assert!(range.start >= 2);
        DecompositionLevelCount(random_usize_between(range))
    }

    pub fn random_i32_between(range: std::ops::Range<i32>) -> i32 {
        use rand::distributions::{Distribution, Uniform};
        let between = Uniform::from(range);
        let mut rng = rand::thread_rng();
        between.sample(&mut rng)
    }

    pub fn random_usize_between(range: std::ops::Range<usize>) -> usize {
        assert!(range.end > range.start);
        assert_ne!(range.end, 0);
        use rand::distributions::{Distribution, Uniform};
        let between = Uniform::from(range);
        let mut rng = rand::thread_rng();
        between.sample(&mut rng)
    }

    pub fn any_usize() -> usize {
        random_usize_between(0..usize::MAX)
    }

    pub fn random_utorus_between<T: UnsignedTorus>(range: std::ops::Range<T>) -> T {
        let val: T = random_uniform();
        val % (range.end - range.start) + range.start
    }

    pub fn any_utorus<T: UnsignedTorus>() -> T {
        random_uniform()
    }
}
