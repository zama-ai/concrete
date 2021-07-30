use std::fmt::Debug;

use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
};
use concrete_fftw::array::AlignedVec;
use serde::{Deserialize, Serialize};

use crate::backends::core::private::{
    crypto::ggsw::FourierGgswSeededCiphertext,
    math::{
        fft::{Complex64, FourierPolynomial},
        random::RandomGenerator,
        tensor::{
            ck_dim_div, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
        },
        torus::UnsignedTorus,
    },
};

use super::{FourierBootstrapKey, FourierBuffers, StandardSeededBootstrapKey};

#[cfg(test)]
mod tests;

/// A bootstrapping key in the fourier domain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    seed: u128,
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::crypto::{GlweSize, LweDimension, LweSize};
    /// use concrete_core::math::decomposition::{DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::polynomial::PolynomialSize;
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
        FourierSeededBootstrapKey {
            tensor,
            decomp_level,
            decomp_base_log,
            glwe_size,
            poly_size,
            _scalar: Default::default(),
            seed: RandomGenerator::generate_u128(),
        }
    }
}

impl<Cont, Scalar> FourierSeededBootstrapKey<Cont, Scalar>
where
    Scalar: UnsignedTorus,
{
    /// Creates a bootstrapping key from an existing container of values.
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        seed: u128,
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
            seed,
        }
    }

    /// Fills a fourier bootstrapping key with the fourier transform of a bootstrapping key in
    /// coefficient domain.
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

    /// Returns an iterator over the borrowed seeded GGSW ciphertext composing the key.
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
        let seed = self.seed;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                FourierGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    seed,
                    i,
                )
            })
    }

    /// Returns an iterator over the mutably borrowed GGSW ciphertext composing the key.
    pub fn ggsw_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierGgswSeededCiphertext<&mut [Complex64], Scalar>>
    where
        Self: AsMutTensor<Element = Complex64>,
    {
        let chunks_size =
            self.glwe_size.0 * self.glwe_size.0 * self.poly_size.0 * self.decomp_level.0;
        let glwe_size = self.glwe_size;
        let poly_size = self.poly_size;
        let base_log = self.decomp_base_log;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(i, tensor)| {
                FourierGgswSeededCiphertext::from_container(
                    tensor.into_container(),
                    glwe_size,
                    poly_size,
                    base_log,
                    seed,
                    i,
                )
            })
    }

    pub fn expand_into<OutCont>(
        &self,
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
