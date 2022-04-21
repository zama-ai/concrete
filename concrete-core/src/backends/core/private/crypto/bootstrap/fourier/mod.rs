use std::fmt::Debug;

use concrete_fftw::array::AlignedVec;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::bootstrap::standard::StandardBootstrapKey;
use crate::backends::core::private::crypto::ggsw::FourierGgswCiphertext;
use crate::backends::core::private::crypto::glwe::GlweCiphertext;
use crate::backends::core::private::crypto::lwe::LweCiphertext;
use crate::backends::core::private::math::fft::Complex64;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LutCountLog, LweDimension,
    ModulusSwitchOffset, MonomialDegree, PolynomialSize,
};

mod buffers;
#[cfg(test)]
mod tests;

pub use buffers::{FftBuffers, FourierBuffers};

/// A bootstrapping key in the fourier domain.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct FourierBootstrapKey<Cont, Scalar>
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
}

impl<Scalar> FourierBootstrapKey<AlignedVec<Complex64>, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Allocates a new complex bootstrapping key whose polynomials coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
            key_size.0 * decomp_level.0 * glwe_size.0 * glwe_size.0 * poly_size.0,
        ));
        tensor.as_mut_tensor().fill_with_element(value);
        FourierBootstrapKey {
            tensor,
            poly_size,
            glwe_size,
            decomp_level,
            decomp_base_log,
            _scalar: Default::default(),
        }
    }
}

impl<Cont, Scalar> FourierBootstrapKey<Cont, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Creates a bootstrapping key from an existing container of values.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let vector = vec![Complex64::new(0., 0.); 256 * 5 * 4 * 4 * 15];
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::from_container(
    ///     vector.as_slice(),
    ///     GlweSize(4),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(5),
    ///     DecompositionBaseLog(4),
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
    ) -> FourierBootstrapKey<Cont, Scalar>
    where
        Cont: AsRefSlice<Element = Complex64>,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() =>
            decomp_level.0,
            glwe_size.0 * glwe_size.0,
            poly_size.0
        );
        FourierBootstrapKey {
            tensor,
            poly_size,
            glwe_size,
            decomp_level,
            decomp_base_log,
            _scalar: Default::default(),
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
    ///     FourierBootstrapKey, FourierBuffers, StandardBootstrapKey,
    /// };
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk = StandardBootstrapKey::allocate(
    ///     9u32,
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// let mut frr_bsk = FourierBootstrapKey::allocate(
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
        coef_bsk: &StandardBootstrapKey<InputCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        StandardBootstrapKey<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We move every GGSW to the fourier domain.
        let iterator = self.ggsw_iter_mut().zip(coef_bsk.ggsw_iter());
        for (mut fourier_ggsw, coef_ggsw) in iterator {
            fourier_ggsw.fill_with_forward_fourier(&coef_ggsw, buffers);
        }
    }

    /// Returns the size of the polynomials used in the bootstrapping key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
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
            self.glwe_size.0 * self.glwe_size.0,
            self.decomp_level.0
        );
        LweDimension(
            self.as_tensor().len()
                / (self.glwe_size.0 * self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0),
        )
    }

    /// Returns an iterator over the borrowed GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// for ggsw in bsk.ggsw_iter() {
    ///     assert_eq!(ggsw.polynomial_size(), PolynomialSize(256));
    ///     assert_eq!(ggsw.glwe_size(), GlweSize(7));
    ///     assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// }
    /// assert_eq!(bsk.ggsw_iter().count(), 4);
    /// ```
    pub fn ggsw_iter(&self) -> impl Iterator<Item = FourierGgswCiphertext<&[Complex64], Scalar>>
    where
        Self: AsRefTensor<Element = Complex64>,
    {
        let chunks_size =
            self.glwe_size.0 * self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let rlwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .map(move |tensor| {
                FourierGgswCiphertext::from_container(
                    tensor.into_container(),
                    rlwe_size,
                    poly_size,
                    base_log,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed GGSW ciphertext composing the key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut bsk: FourierBootstrapKey<_, u32> = FourierBootstrapKey::allocate(
    ///     Complex64::new(9., 8.),
    ///     GlweSize(7),
    ///     PolynomialSize(256),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(5),
    ///     LweDimension(4),
    /// );
    /// for mut ggsw in bsk.ggsw_iter_mut() {
    ///     ggsw.as_mut_tensor()
    ///         .fill_with_element(Complex64::new(0., 0.));
    /// }
    /// assert!(bsk.as_tensor().iter().all(|a| *a == Complex64::new(0., 0.)));
    /// assert_eq!(bsk.ggsw_iter_mut().count(), 4);
    /// ```
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierGgswCiphertext<&mut [Complex64], Scalar>>
    where
        Self: AsMutTensor<Element = Complex64>,
    {
        let chunks_size =
            self.glwe_size.0 * self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let rlwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(move |tensor| {
                FourierGgswCiphertext::from_container(
                    tensor.into_container(),
                    rlwe_size,
                    poly_size,
                    base_log,
                )
            })
    }

    // This cmux mutates both ct1 and ct0. The result is in ct0 after the method was called.
    fn cmux<C0, C1, C2>(
        &self,
        ct0: &mut GlweCiphertext<C0>,
        ct1: &mut GlweCiphertext<C1>,
        ggsw: &FourierGgswCiphertext<C2, Scalar>,
        fft_buffers: &mut FftBuffers,
        rounded_buffer: &mut GlweCiphertext<Vec<Scalar>>,
    ) where
        GlweCiphertext<C0>: AsMutTensor<Element = Scalar>,
        GlweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        FourierGgswCiphertext<C2, Scalar>: AsRefTensor<Element = Complex64>,
        Scalar: UnsignedTorus,
    {
        ct1.as_mut_tensor()
            .update_with_wrapping_sub(ct0.as_tensor());
        ggsw.external_product(ct0, ct1, fft_buffers, rounded_buffer);
    }

    fn blind_rotate<C2>(&self, buffers: &mut FourierBuffers<Scalar>, lwe: &LweCiphertext<C2>)
    where
        LweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<Vec<Scalar>>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Complex64>,
        Scalar: UnsignedTorus,
    {
        // We unpack the lwe ciphertext.
        let (lwe_body, lwe_mask) = lwe.get_body_and_mask();
        let lut = &mut buffers.lut_buffer;

        // We perform the initial clear rotation by performing lut <- lut * X^{-body_hat}
        let lut_poly_size = lut.polynomial_size();
        lut.as_mut_polynomial_list()
            .update_with_wrapping_monic_monomial_div(pbs_modulus_switch(
                lwe_body.0,
                lut_poly_size,
                ModulusSwitchOffset(0),
                LutCountLog(0),
            ));

        // We initialize the ct_0 and ct_1 used for the successive cmuxes
        let ct_0 = lut;
        let mut ct_1 = GlweCiphertext::allocate(Scalar::ZERO, ct_0.polynomial_size(), ct_0.size());

        // We iterate over the bootstrap key elements and perform the blind rotation.
        for (lwe_mask_element, bootstrap_key_ggsw) in
            lwe_mask.mask_element_iter().zip(self.ggsw_iter())
        {
            // We copy ct_0 to ct_1
            ct_1.as_mut_tensor()
                .as_mut_slice()
                .copy_from_slice(ct_0.as_tensor().as_slice());

            // If the mask is not zero, we perform the cmux
            if *lwe_mask_element != Scalar::ZERO {
                // We rotate ct_1 by performing ct_1 <- ct_1 * X^{a_hat}
                ct_1.as_mut_polynomial_list()
                    .update_with_wrapping_monic_monomial_mul(pbs_modulus_switch(
                        *lwe_mask_element,
                        lut_poly_size,
                        ModulusSwitchOffset(0),
                        LutCountLog(0),
                    ));
                // We perform the cmux.
                self.cmux(
                    ct_0,
                    &mut ct_1,
                    &bootstrap_key_ggsw,
                    &mut buffers.fft_buffers,
                    &mut buffers.rounded_buffer,
                );
            }
        }
    }
}

// This function switches modulus for a single coefficient of a ciphertext,
// only in the context of a PBS
//
// offset: the number of msb discarded
// lut_count_log: the right padding
pub fn pbs_modulus_switch<Scalar>(
    input: Scalar,
    poly_size: PolynomialSize,
    offset: ModulusSwitchOffset,
    lut_count_log: LutCountLog,
) -> MonomialDegree
where
    Scalar: UnsignedTorus,
{
    // First, do the left shift (we discard the offset msb)
    let mut output = input << offset.0;
    // Start doing the right shift
    output >>= Scalar::BITS - poly_size.log2().0 - 2 + lut_count_log.0;
    // Do the rounding
    output += output & Scalar::ONE;
    // Finish the right shift
    output >>= 1;
    // Apply the lsb padding
    output <<= lut_count_log.0;
    MonomialDegree(output.cast_into() as usize)
}

impl<Cont, Scalar> FourierBootstrapKey<Cont, Scalar>
where
    GlweCiphertext<Vec<Scalar>>: AsRefTensor<Element = Scalar>,
    Self: AsRefTensor<Element = Complex64>,
    Scalar: UnsignedTorus,
{
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
    /// use concrete_core::backends::core::private::crypto::bootstrap::{
    ///     FourierBootstrapKey, FourierBuffers, StandardBootstrapKey,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::glwe::GlweCiphertext;
    /// use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::{GlweSecretKey, LweSecretKey};
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::AsMutTensor;
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
    ///
    /// let mut buffers = FourierBuffers::new(fourier_bsk.polynomial_size(), fourier_bsk.glwe_size());
    /// fourier_bsk.fill_with_forward_fourier(&coef_bsk, &mut buffers);
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
    /// fourier_bsk.bootstrap(&mut lwe_out, &lwe_in, &accumulator, &mut buffers);
    /// ```
    pub fn bootstrap<C1, C2, C3>(
        &self,
        lwe_out: &mut LweCiphertext<C1>,
        lwe_in: &LweCiphertext<C2>,
        accumulator: &GlweCiphertext<C3>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        LweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        LweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<C3>: AsRefTensor<Element = Scalar>,
    {
        // We retrieve the accumulator buffer, and fill it with the input accumulator values.
        {
            let local_accumulator = &mut buffers.lut_buffer;
            local_accumulator
                .as_mut_tensor()
                .as_mut_slice()
                .copy_from_slice(accumulator.as_tensor().as_slice());
        }

        // We perform the blind rotate
        self.blind_rotate(buffers, lwe_in);

        // We perform the extraction of the first sample.
        let local_accumulator = &mut buffers.lut_buffer;
        local_accumulator.fill_lwe_with_sample_extraction(lwe_out, MonomialDegree(0));
    }
}

impl<Element, Cont, Scalar> AsRefTensor for FourierBootstrapKey<Cont, Scalar>
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

impl<Element, Cont, Scalar> AsMutTensor for FourierBootstrapKey<Cont, Scalar>
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

impl<Cont, Scalar> IntoTensor for FourierBootstrapKey<Cont, Scalar>
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
