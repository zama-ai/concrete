use crate::math::random::{Gaussian, RandomGenerable, RandomGenerator, Uniform};
use crate::math::tensor::AsMutTensor;

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::numeric::UnsignedInteger;
use concrete_commons::parameters::{
    DecompositionLevelCount, GlweDimension, GlweSize, LweDimension, LweSize, PolynomialSize,
};
#[cfg(feature = "multithread")]
use rayon::prelude::*;

/// A random number generator which can be used to encrypt messages.
pub struct EncryptionRandomGenerator {
    // A separate mask generator, only used to generate the mask elements.
    mask: RandomGenerator,
    // A separate noise generator, only used to generate the noise elements.
    noise: RandomGenerator,
}

impl EncryptionRandomGenerator {
    /// Creates a new encryption, optionally seeding it with the given value.
    pub fn new(seed: Option<u128>) -> EncryptionRandomGenerator {
        EncryptionRandomGenerator {
            mask: RandomGenerator::new(seed),
            noise: RandomGenerator::new(None),
        }
    }

    // Allows to seed the noise generator. For testing purpose only.
    #[allow(dead_code)]
    pub(crate) fn seed_noise_generator(&mut self, seed: u128) {
        println!("WARNING: The noise generator of the encryption random generator was seeded.");
        self.noise = RandomGenerator::new(Some(seed));
    }

    /// Returns the number of remaining bytes.
    pub fn remaining_bytes(&self) -> u128 {
        self.mask.remaining_bytes()
    }

    // Forks the generator, when splitting a bootstrap key into ggsw ct.
    #[allow(dead_code)]
    pub(crate) fn fork_bsk_to_ggsw<T: UnsignedInteger>(
        &mut self,
        lwe_dimension: LweDimension,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_ggsw::<T>(level, glwe_size, polynomial_size);
        self.try_fork(lwe_dimension.0, mask_bytes)
    }

    // Forks the generator into a parallel iterator, when splitting a bootstrap key into ggsw ct.
    #[cfg(feature = "multithread")]
    pub(crate) fn par_fork_bsk_to_ggsw<T: UnsignedInteger>(
        &mut self,
        lwe_dimension: LweDimension,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_ggsw::<T>(level, glwe_size, polynomial_size);
        self.par_try_fork(lwe_dimension.0, mask_bytes)
    }

    // Forks the generator, when splitting a ggsw into level matrices.
    pub(crate) fn fork_ggsw_to_ggsw_levels<T: UnsignedInteger>(
        &mut self,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_ggsw_level::<T>(glwe_size, polynomial_size);
        self.try_fork(level.0, mask_bytes)
    }

    // Forks the generator into a parallel iterator, when splitting a ggsw into level matrices.
    #[cfg(feature = "multithread")]
    pub(crate) fn par_fork_ggsw_to_ggsw_levels<T: UnsignedInteger>(
        &mut self,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_ggsw_level::<T>(glwe_size, polynomial_size);
        self.par_try_fork(level.0, mask_bytes)
    }

    // Forks the generator, when splitting a ggsw level matrix to glwe.
    pub(crate) fn fork_ggsw_level_to_glwe<T: UnsignedInteger>(
        &mut self,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_glwe::<T>(glwe_size.to_glwe_dimension(), polynomial_size);
        self.try_fork(glwe_size.0, mask_bytes)
    }

    // Forks the generator into a parallel iterator, when splitting a ggsw level matrix to glwe.
    #[cfg(feature = "multithread")]
    pub(crate) fn par_fork_ggsw_level_to_glwe<T: UnsignedInteger>(
        &mut self,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_glwe::<T>(glwe_size.to_glwe_dimension(), polynomial_size);
        self.par_try_fork(glwe_size.0, mask_bytes)
    }

    // Forks the generator, when splitting a ggsw into level matrices.
    pub(crate) fn fork_gsw_to_gsw_levels<T: UnsignedInteger>(
        &mut self,
        level: DecompositionLevelCount,
        lwe_size: LweSize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_gsw_level::<T>(lwe_size);
        self.try_fork(level.0, mask_bytes)
    }

    // Forks the generator into a parallel iterator, when splitting a ggsw into level matrices.
    #[cfg(feature = "multithread")]
    pub(crate) fn par_fork_gsw_to_gsw_levels<T: UnsignedInteger>(
        &mut self,
        level: DecompositionLevelCount,
        lwe_size: LweSize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_gsw_level::<T>(lwe_size);
        self.par_try_fork(level.0, mask_bytes)
    }

    // Forks the generator, when splitting a ggsw level matrix to glwe.
    pub(crate) fn fork_gsw_level_to_lwe<T: UnsignedInteger>(
        &mut self,
        lwe_size: LweSize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_lwe::<T>(lwe_size.to_lwe_dimension());
        self.try_fork(lwe_size.0, mask_bytes)
    }

    // Forks the generator into a parallel iterator, when splitting a ggsw level matrix to glwe.
    #[cfg(feature = "multithread")]
    pub(crate) fn par_fork_gsw_level_to_lwe<T: UnsignedInteger>(
        &mut self,
        lwe_size: LweSize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        let mask_bytes = mask_bytes_per_lwe::<T>(lwe_size.to_lwe_dimension());
        self.par_try_fork(lwe_size.0, mask_bytes)
    }

    // Forks both generators into an iterator
    fn try_fork(
        &mut self,
        n_child: usize,
        mask_bytes: usize,
    ) -> Option<impl Iterator<Item = EncryptionRandomGenerator>> {
        // We try to fork the generators
        let mask_iter = self.mask.try_sequential_fork(n_child, mask_bytes)?;
        let noise_iter = self.noise.try_alternate_fork(n_child)?;

        // We return a proper iterator.
        Some(
            mask_iter
                .zip(noise_iter)
                .map(|(mask, noise)| EncryptionRandomGenerator { mask, noise }),
        )
    }

    // Forks both generators into a parallel iterator.
    #[cfg(feature = "multithread")]
    fn par_try_fork(
        &mut self,
        n_child: usize,
        mask_bytes: usize,
    ) -> Option<impl IndexedParallelIterator<Item = EncryptionRandomGenerator>> {
        // We try to fork the generators
        let mask_iter = self.mask.par_try_sequential_fork(n_child, mask_bytes)?;
        let noise_iter = self.noise.par_try_alternate_fork(n_child)?;

        // We return a proper iterator.
        Some(
            mask_iter
                .zip(noise_iter)
                .map(|(mask, noise)| EncryptionRandomGenerator { mask, noise }),
        )
    }

    // Fills the tensor with random uniform values, using the mask generator.
    pub(crate) fn fill_tensor_with_random_mask<Scalar, Tensorable>(
        &mut self,
        output: &mut Tensorable,
    ) where
        Scalar: RandomGenerable<Uniform>,
        Tensorable: AsMutTensor<Element = Scalar>,
    {
        self.mask.fill_tensor_with_random_uniform(output)
    }

    // Sample a noise value, using the noise generator.
    pub(crate) fn random_noise<Scalar>(&mut self, std: impl DispersionParameter) -> Scalar
    where
        Scalar: RandomGenerable<Gaussian<f64>>,
    {
        <Scalar>::generate_one(
            &mut self.noise,
            Gaussian {
                std: std.get_standard_dev(),
                mean: 0.,
            },
        )
    }

    // Fills the input tensor with random noise, using the noise generator.
    pub(crate) fn fill_tensor_with_random_noise<Scalar, Tensorable>(
        &mut self,
        output: &mut Tensorable,
        std: impl DispersionParameter,
    ) where
        (Scalar, Scalar): RandomGenerable<Gaussian<f64>>,
        Tensorable: AsMutTensor<Element = Scalar>,
    {
        self.noise
            .fill_tensor_with_random_gaussian(output, 0., std.get_standard_dev());
    }
}

fn mask_bytes_per_coef<T: UnsignedInteger>() -> usize {
    T::BITS / 8
}

fn mask_bytes_per_polynomial<T: UnsignedInteger>(poly_size: PolynomialSize) -> usize {
    poly_size.0 * mask_bytes_per_coef::<T>()
}

fn mask_bytes_per_glwe<T: UnsignedInteger>(
    glwe_dimension: GlweDimension,
    poly_size: PolynomialSize,
) -> usize {
    glwe_dimension.0 * mask_bytes_per_polynomial::<T>(poly_size)
}

fn mask_bytes_per_ggsw_level<T: UnsignedInteger>(
    glwe_size: GlweSize,
    poly_size: PolynomialSize,
) -> usize {
    glwe_size.0 * mask_bytes_per_glwe::<T>(glwe_size.to_glwe_dimension(), poly_size)
}

fn mask_bytes_per_lwe<T: UnsignedInteger>(lwe_dimension: LweDimension) -> usize {
    lwe_dimension.0 * mask_bytes_per_coef::<T>()
}

fn mask_bytes_per_gsw_level<T: UnsignedInteger>(lwe_size: LweSize) -> usize {
    lwe_size.0 * mask_bytes_per_lwe::<T>(lwe_size.to_lwe_dimension())
}

fn mask_bytes_per_ggsw<T: UnsignedInteger>(
    level: DecompositionLevelCount,
    glwe_size: GlweSize,
    poly_size: PolynomialSize,
) -> usize {
    level.0 * mask_bytes_per_ggsw_level::<T>(glwe_size, poly_size)
}
