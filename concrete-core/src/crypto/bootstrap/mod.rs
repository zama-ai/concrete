//! Bootstrapping keys.
//!
//! The bootstrapping operation allows to reduce the level of noise in an LWE ciphertext, while
//! evaluating an univariate function.

pub use fourier::FourierBootstrapKey;
pub use standard::StandardBootstrapKey;

use crate::crypto::glwe::GlweCiphertext;
use crate::crypto::lwe::LweCiphertext;
use crate::math::tensor::{AsMutTensor, AsRefTensor};
use crate::math::torus::UnsignedTorus;

mod fourier;
mod standard;
mod surrogate;

/// A trait for bootstrap keys types performing a bootstrap operation.
pub trait Bootstrap {
    /// The types of data used in the bootstrapped ciphertexts.
    type CiphertextScalar: UnsignedTorus;

    /// Performs a bootstrap of an lwe ciphertext, with a given accumulator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::numeric::CastInto;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::crypto::bootstrap::{Bootstrap, FourierBootstrapKey, StandardBootstrapKey};
    /// use concrete_core::crypto::encoding::Plaintext;
    /// use concrete_core::crypto::glwe::GlweCiphertext;
    /// use concrete_core::crypto::lwe::LweCiphertext;
    /// use concrete_core::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::tensor::AsMutTensor;
    ///
    /// // define settings
    /// let polynomial_size = PolynomialSize(1024);
    /// let rlwe_dimension = GlweDimension(1);
    /// let lwe_dimension = LweDimension(630);
    ///
    /// let level = DecompositionLevelCount(3);
    /// let base_log = DecompositionBaseLog(7);
    /// let std = LogStandardDev::from_log_standard_dev(-29.);
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    ///
    /// let mut rlwe_sk =
    ///     GlweSecretKey::generate_binary(rlwe_dimension, polynomial_size, &mut secret_generator);
    /// let mut lwe_sk = LweSecretKey::generate_binary(lwe_dimension, &mut secret_generator);
    ///
    /// // allocation and generation of the key in coef domain:
    /// let mut coef_bsk = StandardBootstrapKey::allocate(
    ///     0 as u32,
    ///     rlwe_dimension.to_glwe_size(),
    ///     polynomial_size,
    ///     level,
    ///     base_log,
    ///     lwe_dimension,
    /// );
    /// coef_bsk.fill_with_new_key(&lwe_sk, &rlwe_sk, std, &mut encryption_generator);
    ///
    /// // allocation for the bootstrapping key
    /// let mut fourier_bsk = FourierBootstrapKey::allocate(
    ///     Complex64::new(0., 0.),
    ///     rlwe_dimension.to_glwe_size(),
    ///     polynomial_size,
    ///     level,
    ///     base_log,
    ///     lwe_dimension,
    /// );
    /// fourier_bsk.fill_with_forward_fourier(&coef_bsk);
    ///
    /// let message = Plaintext(2u32.pow(30));
    ///
    /// let mut lwe_in = LweCiphertext::allocate(0u32, lwe_dimension.to_lwe_size());
    /// let mut lwe_out =
    ///     LweCiphertext::allocate(0u32, LweSize(rlwe_dimension.0 * polynomial_size.0 + 1));
    /// lwe_sk.encrypt_lwe(&mut lwe_in, &message, std, &mut encryption_generator);
    ///
    /// // accumulator is a trivial encryption of [0, 1/2N, 2/2N, ...]
    /// let mut accumulator =
    ///     GlweCiphertext::allocate(0u32, polynomial_size, rlwe_dimension.to_glwe_size());
    /// accumulator
    ///     .get_mut_body()
    ///     .as_mut_tensor()
    ///     .iter_mut()
    ///     .enumerate()
    ///     .for_each(|(i, a)| {
    ///         *a = (i as f64 * 2_f64.powi(32_i32 - 10 - 1)).cast_into();
    ///     });
    ///
    /// // bootstrap
    /// fourier_bsk.bootstrap(&mut lwe_out, &lwe_in, &accumulator);
    /// ```
    fn bootstrap<C1, C2, C3>(
        &self,
        lwe_out: &mut LweCiphertext<C1>,
        lwe_in: &LweCiphertext<C2>,
        accumulator: &GlweCiphertext<C3>,
    ) where
        LweCiphertext<C1>: AsMutTensor<Element = Self::CiphertextScalar>,
        LweCiphertext<C2>: AsRefTensor<Element = Self::CiphertextScalar>,
        GlweCiphertext<C3>: AsRefTensor<Element = Self::CiphertextScalar>;
}

#[cfg(all(test, feature = "multithread"))]
mod test {
    use crate::crypto::bootstrap::StandardBootstrapKey;
    use crate::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
    use crate::crypto::secret::{GlweSecretKey, LweSecretKey};
    use crate::math::torus::UnsignedTorus;
    use concrete_commons::dispersion::StandardDev;
    use concrete_commons::parameters::{
        DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    };

    fn test_bsk_gen_equivalence<T: UnsignedTorus + Send + Sync>() {
        for _ in 0..10 {
            let lwe_dim = LweDimension(crate::test_tools::random_usize_between(5..10));
            let glwe_dim = GlweDimension(crate::test_tools::random_usize_between(5..10));
            let poly_size = PolynomialSize(crate::test_tools::random_usize_between(5..10));
            let level = DecompositionLevelCount(crate::test_tools::random_usize_between(2..5));
            let base_log = DecompositionBaseLog(crate::test_tools::random_usize_between(2..5));
            let mask_seed = crate::test_tools::any_usize() as u128;
            let noise_seed = crate::test_tools::any_usize() as u128;

            let mut secret_generator = SecretRandomGenerator::new(None);
            let lwe_sk = LweSecretKey::generate_binary(lwe_dim, &mut secret_generator);
            let glwe_sk =
                GlweSecretKey::generate_binary(glwe_dim, poly_size, &mut secret_generator);

            let mut mono_bsk = StandardBootstrapKey::allocate(
                T::ZERO,
                glwe_dim.to_glwe_size(),
                poly_size,
                level,
                base_log,
                lwe_dim,
            );
            let mut encryption_generator = EncryptionRandomGenerator::new(Some(mask_seed));
            encryption_generator.seed_noise_generator(noise_seed);
            mono_bsk.fill_with_new_key(
                &lwe_sk,
                &glwe_sk,
                StandardDev::from_standard_dev(10.),
                &mut encryption_generator,
            );

            let mut multi_bsk = StandardBootstrapKey::allocate(
                T::ZERO,
                glwe_dim.to_glwe_size(),
                poly_size,
                level,
                base_log,
                lwe_dim,
            );
            let mut encryption_generator = EncryptionRandomGenerator::new(Some(mask_seed));
            encryption_generator.seed_noise_generator(noise_seed);
            multi_bsk.par_fill_with_new_key(
                &lwe_sk,
                &glwe_sk,
                StandardDev::from_standard_dev(10.),
                &mut encryption_generator,
            );

            assert_eq!(mono_bsk, multi_bsk);
        }
    }

    #[test]
    fn test_bsk_gen_equivalence_u32() {
        test_bsk_gen_equivalence::<u32>()
    }

    #[test]
    fn test_bsk_gen_equivalence_u64() {
        test_bsk_gen_equivalence::<u64>()
    }
}
