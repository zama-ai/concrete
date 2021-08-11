use std::cell::RefCell;
use std::fmt::Debug;

use concrete_fftw::array::AlignedVec;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::surrogate::SurrogateBsk;
use crate::crypto::bootstrap::standard::StandardBootstrapKey;
use crate::crypto::bootstrap::surrogate::BskKind;
use crate::crypto::bootstrap::Bootstrap;
use crate::crypto::ggsw::GgswCiphertext;
use crate::crypto::glwe::GlweCiphertext;
use crate::crypto::lwe::LweCiphertext;
use crate::math::decomposition::SignedDecomposer;
use crate::math::fft::{Complex64, Fft, FourierPolynomial};
use crate::math::polynomial::{MonomialDegree, Polynomial, PolynomialList};
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor};
use crate::math::torus::UnsignedTorus;
use crate::{ck_dim_div, ck_dim_eq, zip, zip_args};
use concrete_commons::numeric::{CastInto, Numeric};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
};
use std::marker::PhantomData;

#[cfg(test)]
mod tests;

const VERSION_STRING: &str = "v0";

/// A bootstrapping key in the fourier domain.
#[derive(Debug, Clone)]
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
    // The fft plan is stored here. This way, we don't pay the price of allocating it every
    // time we need to bootstrap with the same key.
    fft: Fft,
    // The buffers used to perform the fft are also stored in the bootstrap key. Again, the same
    // logic apply, and we don't have to allocate them multiple times.
    fft_first_buffer: RefCell<FourierPolynomial<AlignedVec<Complex64>>>,
    fft_second_buffer: RefCell<FourierPolynomial<AlignedVec<Complex64>>>,
    fft_output_buffer: RefCell<Tensor<AlignedVec<Complex64>>>,
    // Those buffers are also used to store the lut and the rounded input during the bootstrap.
    lut_buffer: RefCell<GlweCiphertext<Vec<Scalar>>>,
    rounded_buffer: RefCell<GlweCiphertext<Vec<Scalar>>>,
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
        let fft = Fft::new(poly_size);
        let fft_first_buffer = RefCell::new(FourierPolynomial::allocate(
            Complex64::new(0., 0.),
            poly_size,
        ));
        let fft_second_buffer = RefCell::new(FourierPolynomial::allocate(
            Complex64::new(0., 0.),
            poly_size,
        ));
        let fft_output_buffer = RefCell::new(Tensor::from_container(AlignedVec::new(
            poly_size.0 * glwe_size.0,
        )));
        let lut_buffer = RefCell::new(GlweCiphertext::allocate(Scalar::ZERO, poly_size, glwe_size));
        let rounded_buffer =
            RefCell::new(GlweCiphertext::allocate(Scalar::ZERO, poly_size, glwe_size));
        FourierBootstrapKey {
            tensor,
            poly_size,
            glwe_size,
            decomp_level,
            decomp_base_log,
            fft,
            fft_first_buffer,
            fft_second_buffer,
            fft_output_buffer,
            lut_buffer,
            rounded_buffer,
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
        let fft = Fft::new(poly_size);
        let fft_first_buffer = RefCell::new(FourierPolynomial::allocate(
            Complex64::new(0., 0.),
            poly_size,
        ));
        let fft_second_buffer = RefCell::new(FourierPolynomial::allocate(
            Complex64::new(0., 0.),
            poly_size,
        ));
        let fft_output_buffer = RefCell::new(Tensor::from_container(AlignedVec::new(
            poly_size.0 * glwe_size.0,
        )));
        let lut_buffer = RefCell::new(GlweCiphertext::allocate(Scalar::ZERO, poly_size, glwe_size));
        let rounded_buffer =
            RefCell::new(GlweCiphertext::allocate(Scalar::ZERO, poly_size, glwe_size));
        FourierBootstrapKey {
            tensor,
            poly_size,
            glwe_size,
            decomp_level,
            decomp_base_log,
            fft,
            fft_first_buffer,
            fft_second_buffer,
            fft_output_buffer,
            lut_buffer,
            rounded_buffer,
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
    /// use concrete_core::crypto::bootstrap::{FourierBootstrapKey, StandardBootstrapKey};
    /// use concrete_core::math::fft::Complex64;
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
    /// frr_bsk.fill_with_forward_fourier(&bsk);
    /// ```
    pub fn fill_with_forward_fourier<InputCont>(
        &mut self,
        coef_bsk: &StandardBootstrapKey<InputCont>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        StandardBootstrapKey<InputCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut *self.fft_first_buffer.borrow_mut();

        // We move every polynomials to the fourier domain.
        let iterator = self
            .tensor
            .subtensor_iter_mut(self.poly_size.0)
            .map(|t| FourierPolynomial::from_container(t.into_container()))
            .zip(coef_bsk.poly_iter());
        for (mut fourier_poly, coef_poly) in iterator {
            self.fft.forward_as_torus(fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one(fft_buffer.as_tensor(), |a| *a);
        }
    }

    // Returns a surrogate bsk from the current one.
    fn as_surrogate(&self) -> SurrogateBsk<&[Complex64], Scalar>
    where
        Self: AsRefTensor<Element = Complex64>,
    {
        SurrogateBsk {
            kind: BskKind::Fourier,
            version: String::from(VERSION_STRING),
            tensor: self.as_tensor().get_sub(..),
            poly_size: self.poly_size,
            glwe_size: self.glwe_size,
            decomp_level: self.decomp_level,
            decomp_base_log: self.decomp_base_log,
            ciphertext_scalar: PhantomData,
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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

    /// Returns the number of levels used to decompose the key bits.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, LweDimension, PolynomialSize,
    /// };
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
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
    pub fn ggsw_iter(&self) -> impl Iterator<Item = GgswCiphertext<&[Complex64]>>
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
                GgswCiphertext::from_container(
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
    /// use concrete_core::crypto::bootstrap::FourierBootstrapKey;
    /// use concrete_core::math::fft::Complex64;
    /// use concrete_core::math::tensor::{AsMutTensor, AsRefTensor};
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
    pub fn ggsw_iter_mut(&mut self) -> impl Iterator<Item = GgswCiphertext<&mut [Complex64]>>
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
                GgswCiphertext::from_container(
                    tensor.into_container(),
                    rlwe_size,
                    poly_size,
                    base_log,
                )
            })
    }

    fn external_product<C1, C2, C3>(
        &self,
        output: &mut GlweCiphertext<C1>,
        ggsw: &GgswCiphertext<C2>,
        glwe: &GlweCiphertext<C3>,
    ) where
        GlweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        GgswCiphertext<C2>: AsRefTensor<Element = Complex64>,
        GlweCiphertext<C3>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We check that the polynomial sizes match
        ck_dim_eq!(
            self.poly_size =>
            glwe.polynomial_size(),
            ggsw.polynomial_size(),
            output.polynomial_size()
        );
        // We check that the glwe sizes match
        ck_dim_eq!(
            self.glwe_size =>
            glwe.size(),
            ggsw.glwe_size(),
            output.size()
        );

        // We mutably borrow the fft buffers. Note that this incurs a (small) cost, which is why
        // we only perform it once and keep the value around. We reset the output buffer.
        let first_fft_buffer = &mut *self.fft_first_buffer.borrow_mut();
        let second_fft_buffer = &mut *self.fft_second_buffer.borrow_mut();
        let output_fft_buffer = &mut *self.fft_output_buffer.borrow_mut();
        output_fft_buffer.fill_with_element(Complex64::new(0., 0.));

        // We also mutably borrow a standard domain buffer to store the rounded input.
        let rounded_input_glwe = &mut *self.rounded_buffer.borrow_mut();

        // We round the input mask and body
        let decomposer = SignedDecomposer::new(self.decomp_base_log, self.decomp_level);
        decomposer.fill_tensor_with_closest_representable(rounded_input_glwe, glwe);

        // ------------------------------------------------------ EXTERNAL PRODUCT IN FOURIER DOMAIN
        // In this section, we perform the external product in the fourier domain, and accumulate
        // the result in the output_fft_buffer variable.
        let mut decomposition = decomposer.decompose_tensor(rounded_input_glwe);
        // We loop through the levels (we reverse to match the order of the decomposition iterator.)
        for ggsw_decomp_matrix in ggsw.level_matrix_iter().rev() {
            // We retrieve the decomposition of this level.
            let glwe_decomp_term = decomposition.next_term().unwrap();
            debug_assert_eq!(
                ggsw_decomp_matrix.decomposition_level(),
                glwe_decomp_term.level()
            );
            // For each levels we have to add the result of the vector-matrix product between the
            // decomposition of the glwe, and the ggsw level matrix to the output. To do so, we
            // iteratively add to the output, the product between every lines of the matrix, and
            // the corresponding (scalar) polynomial in the glwe decomposition:
            //
            //                ggsw_mat                        ggsw_mat
            //   glwe_dec   | - - - - | <        glwe_dec   | - - - - |
            //  | - - - | x | - - - - |         | - - - | x | - - - - | <
            //    ^         | - - - - |             ^       | - - - - |
            //
            //        t = 1                           t = 2                     ...
            // When possible we iterate two times in a row, to benefit from the fact that fft can
            // transform two polynomials at once.
            let mut iterator = zip!(
                ggsw_decomp_matrix.row_iter(),
                glwe_decomp_term
                    .as_tensor()
                    .subtensor_iter(self.poly_size.0)
                    .map(Polynomial::from_tensor)
            );

            //---------------------------------------------------------------- VECTOR-MATRIX PRODUCT
            loop {
                match (iterator.next(), iterator.next()) {
                    // Two iterates are available, we use the fast fft.
                    (Some(first), Some(second)) => {
                        // We unpack the iterator values
                        let zip_args!(first_ggsw_row, first_glwe_poly) = first;
                        let zip_args!(second_ggsw_row, second_glwe_poly) = second;
                        // We perform the forward fft transform for the glwe polynomials
                        self.fft.forward_two_as_integer(
                            first_fft_buffer,
                            second_fft_buffer,
                            &first_glwe_poly,
                            &second_glwe_poly,
                        );
                        // Now we loop through the polynomials of the output, and add the
                        // corresponding product of polynomials.
                        let iterator = zip!(
                            first_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            second_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            output_fft_buffer
                                .as_mut_tensor()
                                .subtensor_iter_mut(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor)
                        );
                        for zip_args!(first_ggsw_poly, second_ggsw_poly, mut output_poly) in
                            iterator
                        {
                            output_poly.update_with_two_multiply_accumulate(
                                &first_ggsw_poly,
                                first_fft_buffer,
                                &second_ggsw_poly,
                                second_fft_buffer,
                            );
                        }
                    }
                    // We reach the  end of the loop and one element remains.
                    (Some(first), None) => {
                        // We unpack the iterator values
                        let (first_ggsw_row, first_glwe_poly) = first;
                        // We perform the forward fft transform for the glwe polynomial
                        self.fft
                            .forward_as_integer(first_fft_buffer, &first_glwe_poly);
                        // Now we loop through the polynomials of the output, and add the
                        // corresponding product of polynomials.
                        let iterator = zip!(
                            first_ggsw_row
                                .as_tensor()
                                .subtensor_iter(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor),
                            output_fft_buffer
                                .subtensor_iter_mut(self.poly_size.0)
                                .map(FourierPolynomial::from_tensor)
                        );
                        for zip_args!(first_ggsw_poly, mut output_poly) in iterator {
                            output_poly.update_with_multiply_accumulate(
                                &first_ggsw_poly,
                                first_fft_buffer,
                            );
                        }
                    }
                    // The loop is over, we can exit.
                    _ => break,
                }
            }
        }

        // --------------------------------------------  TRANSFORMATION OF RESULT TO STANDARD DOMAIN
        // In this section, we bring the result from the fourier domain, back to the standard
        // domain, and add it to the output.
        //
        // We iterate over the polynomials in the output. Again, when possible, we process two
        // iterations simultaneously to benefit from the fft acceleration.
        let mut _output_bind = output.as_mut_polynomial_list();
        let mut iterator = zip!(
            _output_bind.polynomial_iter_mut(),
            output_fft_buffer
                .subtensor_iter_mut(self.poly_size.0)
                .map(FourierPolynomial::from_tensor)
        );
        loop {
            match (iterator.next(), iterator.next()) {
                (Some(first), Some(second)) => {
                    // We unpack the iterates
                    let zip_args!(mut first_output, mut first_fourier) = first;
                    let zip_args!(mut second_output, mut second_fourier) = second;
                    // We perform the backward transform
                    self.fft.add_backward_two_as_torus(
                        &mut first_output,
                        &mut second_output,
                        &mut first_fourier,
                        &mut second_fourier,
                    );
                }
                (Some(first), None) => {
                    // We unpack the iterates
                    let (mut first_output, mut first_fourier) = first;
                    // We perform the backward transform
                    self.fft
                        .add_backward_as_torus(&mut first_output, &mut first_fourier);
                }
                _ => break,
            }
        }
    }

    // This cmux mutates both ct1 and ct0. The result is in ct0 after the method was called.
    fn cmux<C0, C1, C2>(
        &self,
        ct0: &mut GlweCiphertext<C0>,
        ct1: &mut GlweCiphertext<C1>,
        ggsw: &GgswCiphertext<C2>,
    ) where
        GlweCiphertext<C0>: AsMutTensor<Element = Scalar>,
        GlweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        GgswCiphertext<C2>: AsRefTensor<Element = Complex64>,
        Scalar: UnsignedTorus,
    {
        ct1.as_mut_tensor()
            .update_with_wrapping_sub(ct0.as_tensor());
        self.external_product(ct0, ggsw, ct1);
    }

    fn blind_rotate<C1, C2>(&self, lut: &mut GlweCiphertext<C1>, lwe: &LweCiphertext<C2>)
    where
        GlweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        LweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<Vec<Scalar>>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Complex64>,
        Scalar: UnsignedTorus,
    {
        // We unpack the lwe ciphertext.
        let (lwe_body, lwe_mask) = lwe.get_body_and_mask();

        // We define a closure which performs the modulus switching.
        let lut_coef_count: f64 = lut.polynomial_size().0.cast_into();
        let modulus_switch = |input: Scalar| -> usize {
            let tmp: f64 = input.cast_into() / (<Scalar as Numeric>::MAX.cast_into() + 1.);
            let tmp: f64 = tmp * 2. * lut_coef_count;
            let input_hat: usize = tmp.round().cast_into();
            input_hat
        };

        // We perform the initial clear rotation by performing lut <- lut * X^{-body_hat}
        lut.as_mut_polynomial_list()
            .update_with_wrapping_monic_monomial_div(MonomialDegree(modulus_switch(lwe_body.0)));

        // We initialize the ct_0 and ct_1 used for the successive cmuxes
        let mut ct_0 = lut;
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
                    .update_with_wrapping_monic_monomial_mul(MonomialDegree(modulus_switch(
                        *lwe_mask_element,
                    )));
                // We perform the cmux.
                self.cmux(&mut ct_0, &mut ct_1, &bootstrap_key_ggsw);
            }
        }
    }
}

fn constant_sample_extract<LweCont, RlweCont, Scalar>(
    lwe: &mut LweCiphertext<LweCont>,
    glwe: &GlweCiphertext<RlweCont>,
) where
    LweCiphertext<LweCont>: AsMutTensor<Element = Scalar>,
    GlweCiphertext<RlweCont>: AsRefTensor<Element = Scalar>,
    Scalar: UnsignedTorus,
{
    // We extract the mask  and body of both ciphertexts
    let (mut body_lwe, mut mask_lwe) = lwe.get_mut_body_and_mask();
    let (body_glwe, mask_glwe) = glwe.get_body_and_mask();

    // We construct a polynomial list from the lwe mask
    let mut mask_lwe_poly = PolynomialList::from_container(
        mask_lwe.as_mut_tensor().as_mut_slice(),
        glwe.polynomial_size(),
    );

    // We copy the mask values with the proper ordering and sign
    for (mut mask_lwe_polynomial, mask_glwe_polynomial) in mask_lwe_poly
        .polynomial_iter_mut()
        .zip(mask_glwe.as_polynomial_list().polynomial_iter())
    {
        for (lwe_coeff, glwe_coeff) in mask_lwe_polynomial
            .coefficient_iter_mut()
            .zip(mask_glwe_polynomial.coefficient_iter().rev())
        {
            *lwe_coeff = (Scalar::ZERO).wrapping_sub(*glwe_coeff);
        }
    }
    mask_lwe_poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(1));

    // We set the body
    body_lwe.0 = *body_glwe.as_tensor().get_element(0);
}

impl<Cont, Scalar> Bootstrap for FourierBootstrapKey<Cont, Scalar>
where
    GlweCiphertext<Vec<Scalar>>: AsRefTensor<Element = Scalar>,
    Self: AsRefTensor<Element = Complex64>,
    Scalar: UnsignedTorus,
{
    type CiphertextScalar = Scalar;
    fn bootstrap<C1, C2, C3>(
        &self,
        lwe_out: &mut LweCiphertext<C1>,
        lwe_in: &LweCiphertext<C2>,
        accumulator: &GlweCiphertext<C3>,
    ) where
        LweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        LweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        GlweCiphertext<C3>: AsRefTensor<Element = Scalar>,
    {
        // We retrieve the accumulator buffer, and fill it with the input accumulator values.
        let mut local_accumulator = self.lut_buffer.borrow_mut();
        local_accumulator
            .as_mut_tensor()
            .as_mut_slice()
            .copy_from_slice(accumulator.as_tensor().as_slice());
        // We perform the blind rotate
        self.blind_rotate(&mut *local_accumulator, lwe_in);
        // We perform the extraction of the first sample.
        constant_sample_extract(lwe_out, &*local_accumulator);
    }
}

impl<Scalar> Serialize for FourierBootstrapKey<AlignedVec<Complex64>, Scalar>
where
    Scalar: UnsignedTorus,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let surrogate = self.as_surrogate();
        Serialize::serialize(&surrogate, serializer)
    }
}

impl<'de, Scalar> Deserialize<'de> for FourierBootstrapKey<AlignedVec<Complex64>, Scalar>
where
    Scalar: UnsignedTorus,
{
    fn deserialize<D>(
        deserializer: D,
    ) -> Result<FourierBootstrapKey<AlignedVec<Complex64>, Scalar>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let surrogate = <SurrogateBsk<AlignedVec<Complex64>, Scalar> as Deserialize>::deserialize(
            deserializer,
        )?;
        Ok(surrogate.into_fourier_bsk())
    }
}

impl<Cont, Scalar> PartialEq for FourierBootstrapKey<Cont, Scalar>
where
    Cont: PartialEq + AsRefSlice<Element = Complex64>,
    Scalar: UnsignedTorus,
{
    fn eq(&self, other: &Self) -> bool {
        let sur_self = self.as_surrogate();
        let sur_other = other.as_surrogate();
        sur_self.eq(&sur_other)
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
