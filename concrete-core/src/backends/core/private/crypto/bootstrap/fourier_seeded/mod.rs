use std::fmt::Debug;

use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
};
use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
use crate::backends::core::private::math::fft::{Complex64, FourierPolynomial};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{FourierBootstrapKey, FourierBuffers, StandardSeededBootstrapKey};

#[cfg(test)]
mod tests;

/// A bootstrapping key in the fourier domain.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FourierSeededBootstrapKey<Cont, Scalar>
where
    Scalar: UnsignedTorus,
{
    // The tensor containing the actual data of the secret key.
    tensor: Tensor<Cont>,
    // The size of the polynomials
    poly_size: PolynomialSize,
    // The size of the GLWE
    glwe_size: GlweSize,
    // The decomposition parameters
    decomp_level: DecompositionLevelCount,
    decomp_base_log: DecompositionBaseLog,
    _scalar: std::marker::PhantomData<Scalar>,
    seed: Option<Seed>,
}

impl<Scalar> FourierSeededBootstrapKey<AlignedVec<Complex64>, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Allocates a new complex bootstrapping key whose polynomials coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, LweSize,
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(256));
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(3));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(5));
    /// assert_eq!(bsk.key_size(), LweDimension(4));
    /// ```
    pub fn allocate(
        value: Complex64,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        key_size: LweDimension,
    ) -> Self {
        let mut tensor = Tensor::from_container(AlignedVec::new(
            key_size.0 * decomp_level.0 * glwe_size.0 * poly_size.0,
        ));
        tensor.as_mut_tensor().fill_with_element(value);
        FourierSeededBootstrapKey {
            tensor,
            decomp_level,
            decomp_base_log,
            glwe_size,
            poly_size,
            _scalar: Default::default(),
            seed: None,
        }
    }
}

impl<Cont, Scalar> FourierSeededBootstrapKey<Cont, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Creates a bootstrapping key from an existing container of values.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let vector = vec![Complex64::new(0., 0.); 256 * 5 * 4 * 15];
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(4),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(256));
    /// assert_eq!(bsk.glwe_size(), GlweSize(4));
    /// assert_eq!(bsk.level_count(), DecompositionLevelCount(5));
    /// assert_eq!(bsk.base_log(), DecompositionBaseLog(4));
    /// assert_eq!(bsk.key_size(), LweDimension(15));
    /// ```
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice<Element = Complex64>,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() =>
            decomp_level.0,
            glwe_size.0 ,
            poly_size.0
        );
        FourierSeededBootstrapKey {
            tensor,
            decomp_level,
            decomp_base_log,
            glwe_size,
            poly_size,
            _scalar: Default::default(),
            seed: Some(seed),
        }
    }

    /// Fills a fourier bootstrapping key with the fourier transform of a bootstrapping key in
    /// coefficient domain.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::{
    ///     FourierBuffers, FourierSeededBootstrapKey, StandardSeededBootstrapKey,
    /// };
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk = StandardSeededBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// let mut frr_bsk = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(0., 0.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// let mut buffers = FourierBuffers::new(frr_bsk.polynomial_size(), frr_bsk.glwe_size());
    /// frr_bsk.fill_with_forward_fourier(&bsk, &mut buffers);
    /// ```
    pub fn fill_with_forward_fourier<InputCont>(
        &mut self,
        coef_bsk: &StandardSeededBootstrapKey<InputCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        StandardSeededBootstrapKey<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut buffers.fft_buffers.first_buffer;
        let fft = &mut buffers.fft_buffers.fft;

        // We move every polynomials to the fourier domain.
        let iterator = self
            .tensor
            .subtensor_iter_mut(self.poly_size.0)
            .map(|t| FourierPolynomial::from_container(t.into_container()))
            .zip(coef_bsk.poly_iter());
        for (mut fourier_poly, coef_poly) in iterator {
            fft.forward_as_torus(fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one(fft_buffer.as_tensor(), |a| *a);
        }

        self.seed = coef_bsk.get_seed()
    }

    /// Returns the size of the polynomials used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.polynomial_size(), PolynomialSize(256));
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the dimension of the output LWE ciphertext after a bootstrap.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// assert_eq!(bsk.output_lwe_dimension(), LweDimension(1536));
    /// ```
    pub fn output_lwe_dimension(&self) -> LweDimension {
        LweDimension((self.glwe_size.0 - 1) * self.poly_size.0)
    }

    /// Returns the number of levels used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
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

    /// Returns an iterator over the borrowed seeded GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let vector = vec![Complex64::new(0., 0.); 256 * 3 * 7 * 15];
    /// let bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for ggsw in bsk.ggsw_iter() {
    ///     assert_eq!(ggsw.polynomial_size(), PolynomialSize(256));
    ///     assert_eq!(ggsw.glwe_size(), GlweSize(7));
    ///     assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// }
    /// assert_eq!(bsk.ggsw_iter().count(), 15);
    /// ```
    pub fn ggsw_iter(
        &self,
    ) -> impl Iterator<Item = FourierGgswSeededCiphertext<&[Complex64], Scalar>>
    where
        Self: AsRefTensor<Element = Complex64>,
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
                        + i * Scalar::BITS / 8
                            * self.glwe_size().0
                            * self.level_count().0
                            * self.glwe_size().to_glwe_dimension().0
                            * self.polynomial_size().0,
                };
                FourierGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    ggsw_seed,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierSeededBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let vector = vec![Complex64::new(0., 0.); 256 * 5 * 4 * 15];
    /// let mut bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::from_container(
    ///     vector,
    ///     GlweSize(4),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for mut ggsw in bsk.ggsw_iter_mut() {
    ///     ggsw.as_mut_tensor()
    ///         .fill_with_element(Complex64::new(0., 0.));
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a == Complex64::new(0., 0.)));
    /// assert_eq!(bsk.ggsw_iter_mut().count(), 15);
    /// ```
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierGgswSeededCiphertext<&mut [Complex64], Scalar>>
    where
        Self: AsMutTensor<Element = Complex64>,
    {
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
                        + i * Scalar::BITS / 8
                            * glwe_size.0
                            * level_count.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0,
                };
                FourierGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    ggsw_seed,
                )
            })
    }

    /// Returns the ciphertext as a full fledged FourierBootstrapKey
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::{
    ///     FourierBootstrapKey, FourierBuffers, FourierSeededBootstrapKey,
    /// };
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let vector = vec![Complex64::new(0., 0.); 256 * 5 * 4 * 15];
    /// let seeded_bsk: FourierSeededBootstrapKey<_, u32> = FourierSeededBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(4),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// let mut buffers = FourierBuffers::new(PolynomialSize(256), GlweSize(7));
    /// seeded_bsk.expand_into(&mut bsk, &mut buffers);
    /// ```
    pub fn expand_into<OutCont>(
        self,
        output: &mut FourierBootstrapKey<OutCont, Scalar>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Self: AsRefTensor<Element = Complex64>,
        FourierBootstrapKey<OutCont, Scalar>: AsMutTensor<Element = Complex64>,
    {
        output
            .ggsw_iter_mut()
            .zip(self.ggsw_iter())
            .for_each(|(mut ggsw_out, ggsw_in)| {
                ggsw_in.expand_into(&mut ggsw_out, buffers);
            });
    }
}

impl<Element, Cont, Scalar> AsRefTensor for FourierSeededBootstrapKey<Cont, Scalar>
where
    Cont: AsRefSlice<Element = Element>,
    Scalar: UnsignedTorus,
{
    type Element = Element;
    type Container = Cont;
    fn as_tensor(&self) -> &Tensor<Self::Container> {
        &self.tensor
    }
}

impl<Element, Cont, Scalar> AsMutTensor for FourierSeededBootstrapKey<Cont, Scalar>
where
    Cont: AsMutSlice<Element = Element>,
    Scalar: UnsignedTorus,
{
    type Element = Element;
    type Container = Cont;
    fn as_mut_tensor(&mut self) -> &mut Tensor<<Self as AsMutTensor>::Container> {
        &mut self.tensor
    }
}

impl<Cont, Scalar> IntoTensor for FourierSeededBootstrapKey<Cont, Scalar>
where
    Cont: AsRefSlice,
    Scalar: UnsignedTorus,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}
