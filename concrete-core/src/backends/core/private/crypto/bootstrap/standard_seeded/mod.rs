use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::encoding::Plaintext;
use crate::backends::core::private::crypto::ggsw::StandardGgswSeededCiphertext;
use crate::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
use crate::backends::core::private::math::polynomial::Polynomial;
use crate::backends::core::private::math::random::{RandomGenerable, Uniform};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, tensor_traits, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::StandardBootstrapKey;

#[cfg(test)]
mod tests;

/// A seeded bootstrapping key represented in the standard domain.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct StandardSeededBootstrapKey<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    decomp_level: DecompositionLevelCount,
    decomp_base_log: DecompositionBaseLog,
    seed: Option<Seed>,
}

tensor_traits!(StandardSeededBootstrapKey);

impl<Scalar> StandardSeededBootstrapKey<Vec<Scalar>> {
    /// Allocates a new seeded bootstrapping key in the standard domain whose polynomials
    /// coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(9));
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
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
            glwe_size,
            poly_size,
            seed: None,
        }
    }
}

impl<Cont> StandardSeededBootstrapKey<Cont> {
    /// Creates a seeded bootstrapping key from an existing container of values.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let vector = vec![0u32; 10 * 5 * 4 * 15];
    /// let bsk = StandardSeededBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(4),
    ///     PolynomialSize(10),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(10));
    /// assert_eq!(bsk.glwe_size(), GlweSize(4));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(5));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(4));
    /// assert_eq!(bsk.key_size(), LweDimension(15));
    /// ```
    pub fn from_container<Coef>(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        seed: Seed,
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
            seed: Some(seed),
        }
    }

    /// Generate a new seeded bootstrap key from the input parameters, and fills the current
    /// container with it.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    ///     Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    ///
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(9));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let mut bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     glwe_dim.to_glwe_size(),
    ///     poly_size,
    ///     dec_lc,
    ///     dec_bl,
    ///     lwe_dim,
    /// );
    /// let lwe_sk = LweSecretKey::generate_binary(lwe_dim, &mut secret_generator);
    /// let glwe_sk = GlweSecretKey::generate_binary(glwe_dim, poly_size, &mut secret_generator);
    /// bsk.fill_with_new_key(
    ///     &lwe_sk,
    ///     &glwe_sk,
    ///     LogStandardDev::from_log_standard_dev(-15.),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// ```
    pub fn fill_with_new_key<LweCont, GlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<BinaryKeyKind, LweCont>,
        glwe_secret_key: &GlweSecretKey<BinaryKeyKind, GlweCont>,
        noise_parameters: impl DispersionParameter,
        seed: Seed,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, LweCont>: AsRefTensor<Element = Scalar>,
        GlweSecretKey<BinaryKeyKind, GlweCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        ck_dim_eq!(self.key_size().0 => lwe_secret_key.key_size().0);
        self.as_mut_tensor()
            .fill_with_element(<Scalar as Numeric>::ZERO);

        self.seed = Some(seed);

        for (mut ggsw, sk_scalar) in self.ggsw_iter_mut().zip(lwe_secret_key.as_tensor().iter()) {
            let ggsw_seed = ggsw.get_seed().unwrap();
            glwe_secret_key.encrypt_constant_seeded_ggsw(
                &mut ggsw,
                &Plaintext(*sk_scalar),
                noise_parameters,
                ggsw_seed,
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
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
    ///     Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let (lwe_dim, glwe_dim, poly_size) = (LweDimension(4), GlweDimension(6), PolynomialSize(9));
    /// let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(5));
    /// let mut bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     glwe_dim.to_glwe_size(),
    ///     poly_size,
    ///     dec_lc,
    ///     dec_bl,
    ///     lwe_dim,
    /// );
    /// let lwe_sk = LweSecretKey::generate_binary(lwe_dim, &mut secret_generator);
    /// let glwe_sk = GlweSecretKey::generate_binary(glwe_dim, poly_size, &mut secret_generator);
    /// bsk.par_fill_with_new_key(
    ///     &lwe_sk,
    ///     &glwe_sk,
    ///     LogStandardDev::from_log_standard_dev(-15.),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_fill_with_new_key<LweCont, GlweCont, Scalar>(
        &mut self,
        lwe_secret_key: &LweSecretKey<BinaryKeyKind, LweCont>,
        glwe_secret_key: &GlweSecretKey<BinaryKeyKind, GlweCont>,
        noise_parameters: impl DispersionParameter + Sync + Send,
        seed: Seed,
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

        self.seed = Some(seed);

        self.par_ggsw_iter_mut()
            .zip(lwe_secret_key.as_tensor().par_iter())
            .for_each(|(mut ggsw, sk_scalar)| {
                let encoded = Plaintext(*sk_scalar);
                let ggsw_seed = ggsw.get_seed().unwrap();
                glwe_secret_key.par_encrypt_constant_seeded_ggsw(
                    &mut ggsw,
                    &encoded,
                    noise_parameters,
                    ggsw_seed,
                );
            });
    }

    /// Returns the size of the polynomials used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(9));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns the size of the GLWE ciphertexts used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the number of levels used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn level_count(&self) -> DecompositionLevelCount {
        self.decomp_level
    }

    /// Returns the logarithm of the base used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// ```
    pub fn base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns the size of the LWE encrypted key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
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

    pub(crate) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    pub(crate) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    /// Returns an iterator over the borrowed seeded GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    ///
    /// let vector = vec![0u32; 9 * 5 * 7 * 15];
    /// let bsk = StandardSeededBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for ggsw in bsk.ggsw_iter() {
    ///     assert_eq!(ggsw.polynomial_size(), PolynomialSize(9));
    ///     assert_eq!(ggsw.glwe_size(), GlweSize(7));
    ///     assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// }
    /// assert_eq!(bsk.ggsw_iter().count(), 25);
    /// ```
    pub fn ggsw_iter(
        &self,
    ) -> impl Iterator<Item = StandardGgswSeededCiphertext<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let seed = self.seed.unwrap();
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                let ggsw_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + i * <Self as AsRefTensor>::Element::BITS / 8
                            * self.glwe_size().0
                            * self.level_count().0
                            * self.glwe_size().to_glwe_dimension().0
                            * self.polynomial_size().0,
                };
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    ggsw_seed,
                )
            })
    }

    /// Returns a parallel iterator over the mutably borrowed seeded GGSW ciphertext composing the
    /// key.
    ///
    /// # Notes
    ///
    /// This iterator is hidden behind the "multithread" feature gate.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut bsk = StandardSeededBootstrapKey::from_container(
    ///     vec![0u32; 9 * 5 * 7 * 15],
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// bsk.par_ggsw_iter_mut().for_each(|mut ggsw| {
    ///     ggsw.as_mut_tensor().fill_with_element(0);
    /// });
    /// assert!(bsk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(bsk.ggsw_iter_mut().count(), 25);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_ggsw_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<
        Item = StandardGgswSeededCiphertext<&mut [<Self as AsRefTensor>::Element]>,
    >
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric + Sync + Send,
        Cont: Sync + Send,
    {
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let level_count = self.level_count();
        let seed = self.seed.unwrap();

        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                let ggsw_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + i * <Self as AsMutTensor>::Element::BITS / 8
                            * glwe_size.0
                            * level_count.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0,
                };
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    ggsw_seed,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed seeded GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let vector = vec![0u32; 9 * 5 * 4 * 15];
    /// let mut bsk = StandardSeededBootstrapKey::from_container(
    ///     vector,
    ///     GlweSize(4),
    ///     PolynomialSize(9),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for mut ggsw in bsk.ggsw_iter_mut() {
    ///     ggsw.as_mut_tensor().fill_with_element(0);
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(bsk.ggsw_iter_mut().count(), 15);
    /// ```
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = StandardGgswSeededCiphertext<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        // Some(seed + (i * poly_size.0 * decomp_level.0 * glwe_size.0) as u128),
        let chunks_size = self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let level_count = self.level_count();
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                let ggsw_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + i * <Self as AsMutTensor>::Element::BITS / 8
                            * glwe_size.0
                            * level_count.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0,
                };
                StandardGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    ggsw_seed,
                )
            })
    }

    /// Returns an iterator over borrowed polynomials composing the bodies of the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// for poly in bsk.poly_iter() {
    ///     assert_eq!(poly.polynomial_size(), PolynomialSize(256));
    /// }
    /// assert_eq!(bsk.poly_iter().count(), 7 * 3 * 4)
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::StandardSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// for mut poly in bsk.poly_iter_mut() {
    ///     poly.as_mut_tensor().fill_with_element(0u32);
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a == 0));
    /// assert_eq!(bsk.poly_iter_mut().count(), 7 * 3 * 4)
    /// ```
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

    /// Returns the ciphertext as a full fledged StandardBootstrapKey
    ///
    ///  ```
    /// use concrete_commons::dispersion::LogStandardDev;
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize, LweDimension,
    ///     PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::{
    ///     StandardBootstrapKey, StandardSeededBootstrapKey,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::generators::
    /// SecretRandomGenerator; use concrete_core::backends::core::private::crypto::secret::
    /// {GlweSecretKey, LweSecretKey}; use concrete_core::backends::core::private::math::fft::
    /// Complex64; use concrete_core::backends::core::private::math::tensor::{AsMutTensor,
    /// AsRefTensor};
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    ///
    /// // generates a secret key
    /// let glwe_sk = GlweSecretKey::generate_binary(
    ///     GlweDimension(9),
    ///     PolynomialSize(256),
    ///     &mut secret_generator,
    /// );
    ///
    /// let lwe_sk = LweSecretKey::generate_binary(LweDimension(10), &mut secret_generator);
    ///
    /// // allocation and generation of the key in coef domain:
    /// let mut coef_bsk_seeded = StandardSeededBootstrapKey::allocate(
    ///     0u32,
    ///     GlweSize(10),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(10),
    /// );
    /// coef_bsk_seeded.fill_with_new_key(
    ///     &lwe_sk,
    ///     &glwe_sk,
    ///     LogStandardDev::from_log_standard_dev(-20.),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    ///
    /// // expansion of the bootstrapping key
    /// let mut coef_bsk_expanded = StandardBootstrapKey::allocate(
    ///     0u32,
    ///     GlweSize(10),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(10),
    /// );
    /// coef_bsk_seeded.expand_into(&mut coef_bsk_expanded);
    /// ```
    pub fn expand_into<Scalar, OutCont>(self, output: &mut StandardBootstrapKey<OutCont>)
    where
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
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
