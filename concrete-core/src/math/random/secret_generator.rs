use crate::crypto::{GlweDimension, GlweSize, LweDimension};
use crate::math::decomposition::DecompositionLevelCount;
use crate::math::dispersion::DispersionParameter;
use crate::math::polynomial::PolynomialSize;
use crate::math::random::{Gaussian, RandomGenerable, RandomGenerator, Uniform};
use crate::math::tensor::AsMutTensor;
use crate::numeric::UnsignedInteger;

/// A random number generator which can be used to encrypt messages.
pub struct EncryptionRng {
    // A separate mask generator, only used to generate the mask elements.
    mask: RandomGenerator,
    // A separate noise generator, only used to generate the noise elements.
    noise: RandomGenerator,
}

impl EncryptionRng {
    /// Creates a new encryption, optionally seeding it with the given value.
    pub fn new(seed: Option<u128>) -> EncryptionRng {
        EncryptionRng {
            mask: RandomGenerator::new(seed),
            noise: RandomGenerator::new(None),
        }
    }

    /// Returns the number of remaining bytes, if the generator is bounded.
    pub fn remaining_bytes(&self) -> Option<usize> {
        self.mask.remaining_bytes()
    }

    /// Returns whether the generator is bounded.
    pub fn is_bounded(&self) -> bool {
        self.mask.is_bounded()
    }

    pub(crate) fn fork_bsk_to_ggsw<T: UnsignedInteger>(
        &mut self,
        lwe_dimension: LweDimension,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRng>> {
        let mask_bytes = mask_bytes_per_ggsw::<T>(level, glwe_size, polynomial_size);
        let noise_bytes = noise_bytes_per_ggsw(level, glwe_size, polynomial_size);
        self.try_fork(lwe_dimension.0, mask_bytes, noise_bytes)
    }

    pub(crate) fn fork_ggsw_to_ggsw_levels<T: UnsignedInteger>(
        &mut self,
        level: DecompositionLevelCount,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRng>> {
        let mask_bytes = mask_bytes_per_ggsw_level::<T>(glwe_size, polynomial_size);
        let noise_bytes = noise_bytes_per_ggsw_level(glwe_size, polynomial_size);
        self.try_fork(level.0, mask_bytes, noise_bytes)
    }

    pub(crate) fn fork_ggsw_level_to_rlwe<T: UnsignedInteger>(
        &mut self,
        glwe_size: GlweSize,
        polynomial_size: PolynomialSize,
    ) -> Option<impl Iterator<Item = EncryptionRng>> {
        let mask_bytes = mask_bytes_per_glwe::<T>(glwe_size.to_glwe_dimension(), polynomial_size);
        let noise_bytes = noise_bytes_per_glwe(polynomial_size);
        self.try_fork(glwe_size.0, mask_bytes, noise_bytes)
    }

    fn try_fork(
        &mut self,
        n_child: usize,
        mask_bytes: usize,
        noise_bytes: usize,
    ) -> Option<impl Iterator<Item = EncryptionRng>> {
        // We try to fork the generators
        let mask_iter = self.mask.try_fork(n_child, mask_bytes)?;
        let noise_iter = self.noise.try_fork(n_child, noise_bytes)?;

        // We return a proper iterator.
        Some(
            mask_iter
                .zip(noise_iter)
                .map(|(mask, noise)| EncryptionRng { mask, noise }),
        )
    }

    pub(crate) fn random_mask<Scalar: RandomGenerable<Uniform>>(&mut self) -> Scalar {
        self.mask.random_uniform()
    }

    pub(crate) fn fill_tensor_with_random_mask<Scalar, Tensorable>(
        &mut self,
        output: &mut Tensorable,
    ) where
        Scalar: RandomGenerable<Uniform>,
        Tensorable: AsMutTensor<Element = Scalar>,
    {
        self.mask.fill_tensor_with_random_uniform(output)
    }

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

fn mask_bytes_per_ggsw<T: UnsignedInteger>(
    level: DecompositionLevelCount,
    glwe_size: GlweSize,
    poly_size: PolynomialSize,
) -> usize {
    level.0 * mask_bytes_per_ggsw_level::<T>(glwe_size, poly_size)
}

fn noise_bytes_per_coef() -> usize {
    // We use f64 to sample the noise for every precision, and we need 4/pi inputs to generate
    // such an output (we ceil it to 2 to be sure to have enough bytes for each).
    8 * 2
}
fn noise_bytes_per_polynomial(poly_size: PolynomialSize) -> usize {
    poly_size.0 * noise_bytes_per_coef()
}

fn noise_bytes_per_glwe(poly_size: PolynomialSize) -> usize {
    noise_bytes_per_polynomial(poly_size)
}

fn noise_bytes_per_ggsw_level(glwe_size: GlweSize, poly_size: PolynomialSize) -> usize {
    glwe_size.0 * noise_bytes_per_glwe(poly_size)
}

fn noise_bytes_per_ggsw(
    level: DecompositionLevelCount,
    glwe_size: GlweSize,
    poly_size: PolynomialSize,
) -> usize {
    level.0 * noise_bytes_per_ggsw_level(glwe_size, poly_size)
}
