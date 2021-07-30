use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};

use crate::backends::core::private::crypto::encoding::Plaintext;
use crate::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::polynomial::Polynomial;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::StandardBootstrapKey;

#[cfg(test)]
mod tests;

/// A bootstrapping key represented in the standard domain.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardSeededBootstrapKey<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    decomp_level: DecompositionLevelCount,
    decomp_base_log: DecompositionBaseLog,
    seed: u128,
}

tensor_traits!(StandardSeededBootstrapKey);

impl<Scalar> StandardSeededBootstrapKey<Vec<Scalar>> {
    /// Allocates a new seeded bootstrapping key in the standard domain whose polynomials coefficients are
    /// all `value`.
    pub fn allocate(
        value: Scalar,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        key_size: LweDimension,
    ) -> Self
    where
        Scalar: UnsignedTorus,
    {
        StandardSeededBootstrapKey {
            tensor: Tensor::from_container(vec![
                value;
                key_size.0
                    * decomp_level.0
                    * glwe_size.0
                    * poly_size.0
            ]),
            decomp_level,
            decomp_base_log,
            glwe_size: glwe_size,
            poly_size,
            seed: RandomGenerator::generate_u128(),
        }
    }
}

impl<Cont> StandardSeededBootstrapKey<Cont> {
    /// Creates a seeded bootstrapping key from an existing container of values.
    pub fn from_container<Coef>(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        seed: u128,
    ) -> Self
    where
        Cont: AsRefSlice<Element = Coef>,
        Coef: UnsignedTorus,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() =>
            decomp_level.0,
            glwe_size.0,
            poly_size.0
        );
        StandardSeededBootstrapKey {
            tensor,
            glwe_size,
            poly_size,
            decomp_level,
            decomp_base_log,
            seed,
        }
    }

    /// Generate a new seeded bootstrap key from the input parameters, and fills the current container
    /// with it.
    pub fn fill_with_new_key<LweCont, GlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<BinaryKeyKind, LweCont>,
        glwe_secret_key: &GlweSecretKey<BinaryKeyKind, GlweCont>,
        noise_parameters: impl DispersionParameter,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, LweCont>: AsRefTensor<Element = Scalar>,
        GlweSecretKey<BinaryKeyKind, GlweCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.key_size().0 => lwe_secret_key.key_size().0);
        self.as_mut_tensor()
            .fill_with_element(<Scalar as Numeric>::ZERO);

        for (mut ggsw, sk_scalar) in self.ggsw_iter_mut().zip(lwe_secret_key.as_tensor().iter()) {
            glwe_secret_key.encrypt_constant_seeded_ggsw(
                &mut ggsw,
                &Plaintext(*sk_scalar),
                noise_parameters.clone(),
            );
        }
    }

    /// Generate a new bootstrap key from the input parameters, and fills the current container
    /// with it, using all the available threads.
    ///
    /// # Note
    ///
    /// This method uses _rayon_ internally, and is hidden behind the "multithread" feature
    /// gate.
    #[cfg(feature = "multithread")]
    pub fn par_fill_with_new_key<LweCont, GlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<BinaryKeyKind, LweCont>,
        glwe_secret_key: &GlweSecretKey<BinaryKeyKind, GlweCont>,
        noise_parameters: impl DispersionParameter + Sync + Send,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, LweCont>: AsRefTensor<Element = Scalar>,
        GlweSecretKey<BinaryKeyKind, GlweCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus + Sync + Send,
        GlweCont: Sync + Send,
        Cont: Sync + Send,
    {
        ck_dim_eq!(self.key_size().0 => lwe_secret_key.key_size().0);
        self.as_mut_tensor()
            .fill_with_element(<Scalar as Numeric>::ZERO);

        self.par_ggsw_iter_mut()
            .zip(lwe_secret_key.as_tensor().par_iter())
            .for_each(|(mut ggsw, sk_scalar)| {
                let encoded = Plaintext(*sk_scalar);
                glwe_secret_key.par_encrypt_constant_seeded_ggsw(
                    &mut ggsw,
                    &encoded,
                    noise_parameters.clone(),
                );
            });
    }

    /// Returns the size of the polynomials used in the bootstrapping key.
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns the size of the GLWE ciphertexts used in the bootstrapping key.
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the number of levels used to decompose the key bits.
    pub fn level_count(&self) -> DecompositionLevelCount {
        self.decomp_level
    }

    /// Returns the logarithm of the base used to decompose the key bits.
    pub fn base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns the size of the LWE encrypted key.
    pub fn key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.poly_size.0,
            self.glwe_size.0,
            self.decomp_level.0
        );
        LweDimension(
            self.as_tensor().len() / (self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0),
        )
    }

    pub(crate) fn get_seed(&self) -> u128 {
        self.seed
    }

    /// Returns an iterator over the borrowed seeded GGSW ciphertext composing the key.
    pub fn ggsw_iter(
        &self,
    ) -> impl Iterator<Item = StandardGgswSeededCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let seed = self.seed;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    seed,
                    i,
                )
            })
    }

    /// Returns a parallel iterator over the mutably borrowed seeded GGSW ciphertext composing the
    /// key.
    #[cfg(feature = "multithread")]
    pub fn par_ggsw_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<
        Item = StandardGgswSeededCiphertext<&mut [<Self as AsRefTensor>::Element]>,
    >
    where
        Self: AsMutTensor,
        <Self as AsRefTensor>::Element: Sync + Send,
        Cont: Sync + Send,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let seed = self.seed;

        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    seed,
                    i,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed seeded GGSW ciphertext composing the key.
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = StandardGgswSeededCiphertext<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        // Some(seed + (i * poly_size.0 * decomp_level.0 * glwe_size.0) as u128),
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    seed,
                    i,
                )
            })
    }

    /// Returns an iterator over borrowed polynomials composing the bodies of the key.
    pub fn poly_iter(&self) -> impl Iterator<Item = Polynomial<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: UnsignedTorus,
    {
        let poly_size = self.poly_size.0;
        self.as_tensor()
            .subtensor_iter(poly_size)
            .map(|chunk| Polynomial::from_container(chunk.into_container()))
    }

    /// Returns an iterator over mutably borrowed polynomials composing the bodies of the key.
    pub fn poly_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = Polynomial<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: UnsignedTorus,
    {
        let poly_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(poly_size)
            .map(|chunk| Polynomial::from_container(chunk.into_container()))
    }

    pub fn expand_into<Scalar, OutCont>(&self, output: &mut StandardBootstrapKey<OutCont>)
    where
        Scalar: UnsignedTorus,
        StandardBootstrapKey<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
    {
        output
            .ggsw_iter_mut()
            .zip(self.ggsw_iter())
            .for_each(|(mut ggsw_out, ggsw_in)| {
                ggsw_in.expand_into(&mut ggsw_out);
            });
    }
}
