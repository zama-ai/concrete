use crate::backends::core::private::crypto::bootstrap::FourierBuffers;
use crate::backends::core::private::crypto::glwe::{GlweCiphertext, GlweSeededList};
use crate::backends::core::private::math::fft::{Complex64, FourierPolynomial};
use crate::backends::core::private::math::polynomial::Polynomial;
use crate::backends::core::private::math::random::RandomGenerator;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{FourierGgswCiphertext, GgswSeededLevelMatrix, StandardGgswSeededCiphertext};

use crate::backends::core::private::math::decomposition::DecompositionLevel;
use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
};
use concrete_fftw::array::AlignedVec;
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};

/// A GGSW ciphertext in the Fourier Domain.
#[derive(Debug, Clone, PartialEq)]
pub struct FourierGgswSeededCiphertext<Cont, Scalar> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    glwe_size: GlweSize,
    decomp_base_log: DecompositionBaseLog,
    _scalar: std::marker::PhantomData<Scalar>,
    seed: Option<Seed>,
}

impl<Scalar> FourierGgswSeededCiphertext<AlignedVec<Complex64>, Scalar> {
    /// Allocates a new seeded GGSW ciphertext in the Fourier domain whose coefficients are all
    /// `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// assert_eq!(ggsw.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn allocate(
        value: Complex64,
        poly_size: PolynomialSize,
        glwe_size: GlweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Copy,
    {
        let mut tensor =
            Tensor::from_container(AlignedVec::new(decomp_level.0 * glwe_size.0 * poly_size.0));
        tensor.as_mut_tensor().fill_with_element(value);
        FourierGgswSeededCiphertext {
            tensor,
            poly_size,
            glwe_size,
            decomp_base_log,
            _scalar: Default::default(),
            seed: None,
        }
    }
}

impl<Cont, Scalar> FourierGgswSeededCiphertext<Cont, Scalar> {
    /// Creates a seeded GGSW ciphertext in the Fourier domain from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// assert_eq!(ggsw.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn from_container(
        cont: Cont,
        glwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => glwe_size.0, poly_size.0);
        FourierGgswSeededCiphertext {
            tensor,
            poly_size,
            glwe_size,
            decomp_base_log,
            _scalar: Default::default(),
            seed: Some(seed),
        }
    }

    /// Returns the size of the glwe ciphertexts composing the ggsw ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.glwe_size
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.glwe_size.0,
            self.poly_size.0
        );
        DecompositionLevelCount(self.as_tensor().len() / (self.glwe_size.0 * self.poly_size.0))
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns the size of the polynomials used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Returns a borrowed list composed of all the GLWE ciphertext composing current ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let list = ggsw.as_glwe_seeded_list();
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_glwe_seeded_list<E>(&self) -> GlweSeededList<&[E]>
    where
        Self: AsRefTensor<Element = E>,
    {
        GlweSeededList::from_container(
            self.as_tensor().as_slice(),
            self.glwe_size.to_glwe_dimension(),
            self.poly_size,
            self.seed.unwrap(),
        )
    }

    /// Returns a mutably borrowed `GlweList` composed of all the GLWE ciphertext composing
    /// current ciphertext.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension, GlweSize,
    ///     PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let mut ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// let mut list = ggsw.as_mut_glwe_seeded_list();
    /// list.as_mut_tensor()
    ///     .fill_with_element(Complex64::new(0., 0.));
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ggsw.as_tensor()
    ///     .iter()
    ///     .for_each(|a| assert_eq!(*a, Complex64::new(0., 0.)));
    /// ```
    pub fn as_mut_glwe_seeded_list<E>(&mut self) -> GlweSeededList<&mut [E]>
    where
        Self: AsMutTensor<Element = E>,
    {
        let dimension = self.glwe_size.to_glwe_dimension();
        let size = self.poly_size;
        let seed = self.seed.unwrap();
        GlweSeededList::from_container(self.as_mut_tensor().as_mut_slice(), dimension, size, seed)
    }

    /// Returns an iterator over borrowed level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    ///
    /// let ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 9 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(9),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for level_matrix in ggsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     assert_eq!(level_matrix.polynomial_size(), PolynomialSize(9));
    ///     for glwe in level_matrix.row_iter() {
    ///         assert_eq!(glwe.glwe_size(), GlweSize(7));
    ///         assert_eq!(glwe.polynomial_size(), PolynomialSize(9));
    ///     }
    /// }
    /// assert_eq!(ggsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GgswSeededLevelMatrix<&[<Self as AsRefTensor>::Element], Scalar>>
    where
        Self: AsRefTensor,
        Scalar: Numeric,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let matrix_seed = Seed {
                    seed: self.seed.unwrap().seed,
                    shift: self.seed.unwrap().shift
                        + index
                            * self.glwe_size().0
                            * self.glwe_size().to_glwe_dimension().0
                            * self.polynomial_size().0
                            * Scalar::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    matrix_seed,
                )
            })
    }

    /// Returns an iterator over mutably borrowed level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    ///
    /// let mut ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// for mut level_matrix in ggsw.level_matrix_iter_mut() {
    ///     for mut glwe in level_matrix.row_iter_mut() {
    ///         glwe.as_mut_tensor()
    ///             .fill_with_element(Complex64::new(0., 0.));
    ///     }
    /// }
    /// assert!(ggsw
    ///     .as_tensor()
    ///     .iter()
    ///     .all(|a| *a == Complex64::new(0., 0.)));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<
        Item = GgswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element], Scalar>,
    >
    where
        Self: AsMutTensor,
        Scalar: Numeric,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let matrix_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * glwe_size.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0
                            * Scalar::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    matrix_seed,
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed level matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::FourierGgswSeededCiphertext;
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut ggsw: FourierGgswSeededCiphertext<_, u32> = FourierGgswSeededCiphertext::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    ///     Seed { seed: 0, shift: 0 },
    /// );
    /// ggsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut glwe in level_matrix.row_iter_mut() {
    ///             glwe.as_mut_tensor()
    ///                 .fill_with_element(Complex64::new(0., 0.));
    ///         }
    ///     });
    /// assert!(ggsw
    ///     .as_tensor()
    ///     .iter()
    ///     .all(|a| *a == Complex64::new(0., 0.)));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<
        Item = GgswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element], Scalar>,
    >
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
        Scalar: Numeric + Sync + Send,
    {
        let chunks_size = self.poly_size.0 * self.glwe_size.0;
        let poly_size = self.poly_size;
        let glwe_size = self.glwe_size;
        let seed = self.seed.unwrap();
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                let matrix_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + index
                            * glwe_size.0
                            * glwe_size.to_glwe_dimension().0
                            * poly_size.0
                            * Scalar::BITS
                            / 8,
                };
                GgswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    glwe_size,
                    DecompositionLevel(index + 1),
                    matrix_seed,
                )
            })
    }

    /// Fills a GGSW ciphertext with the fourier transform of a GGSW ciphertext in
    /// coefficient domain.
    pub fn fill_with_forward_fourier<InputCont>(
        &mut self,
        coef_ggsw: &StandardGgswSeededCiphertext<InputCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        StandardGgswSeededCiphertext<InputCont>: AsRefTensor<Element = Scalar>,
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
            .zip(
                coef_ggsw
                    .as_tensor()
                    .subtensor_iter(coef_ggsw.polynomial_size().0)
                    .map(|t| Polynomial::from_container(t.into_container())),
            );
        for (mut fourier_poly, coef_poly) in iterator {
            fft.forward_as_torus(fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one((fft_buffer).as_tensor(), |a| *a);
        }

        self.seed = coef_ggsw.get_seed();
    }

    /// Returns the ciphertext as a full fledged FourierGgswCiphertext
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize, Seed,
    /// };
    /// use concrete_core::backends::core::private::crypto::bootstrap::FourierBuffers;
    /// use concrete_core::backends::core::private::crypto::ggsw::{
    ///     FourierGgswCiphertext, FourierGgswSeededCiphertext,
    /// };
    /// use concrete_core::backends::core::private::math::fft::Complex64;
    /// let seeded_ggsw: FourierGgswSeededCiphertext<_, u32> =
    ///     FourierGgswSeededCiphertext::from_container(
    ///         vec![Complex64::new(0., 0.); 7 * 256 * 3],
    ///         GlweSize(7),
    ///         PolynomialSize(256),
    ///         DecompositionBaseLog(4),
    ///         Seed { seed: 0, shift: 0 },
    ///     );
    /// let mut ggsw: FourierGgswCiphertext<_, u32> = FourierGgswCiphertext::allocate(
    ///     Complex64::new(0., 0.),
    ///     PolynomialSize(256),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let mut buffers = FourierBuffers::new(PolynomialSize(256), GlweSize(7));
    /// seeded_ggsw.expand_into(&mut ggsw, &mut buffers);
    /// ```
    pub fn expand_into<OutCont>(
        self,
        output: &mut FourierGgswCiphertext<OutCont, Scalar>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        FourierGgswCiphertext<OutCont, Scalar>: AsMutTensor<Element = Complex64>,
        Self: AsRefTensor<Element = Complex64>,
        Scalar: UnsignedTorus,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut buffers.fft_buffers.first_buffer;
        let fft = &mut buffers.fft_buffers.fft;

        let mut generator = RandomGenerator::new_from_seed(self.seed.unwrap());

        output
            .as_mut_glwe_list()
            .ciphertext_iter_mut()
            .zip(
                self.as_glwe_seeded_list()
                    .complex_ciphertext_iter::<Scalar>(),
            )
            .for_each(|(mut glwe_out, glwe_in)| {
                let mut standard_ct = GlweCiphertext::allocate(
                    Scalar::ZERO,
                    glwe_out.polynomial_size(),
                    glwe_out.size(),
                );
                let mut standard_mask = standard_ct.get_mut_mask();

                // generate a uniformly random mask
                generator.fill_tensor_with_random_uniform(standard_mask.as_mut_tensor());

                // Converts into FFT domain
                for (mut poly_fft, poly_standard) in glwe_out
                    .get_mut_mask()
                    .as_mut_polynomial_list()
                    .polynomial_iter_mut()
                    .zip(standard_mask.as_polynomial_list().polynomial_iter())
                {
                    fft.forward_as_torus(fft_buffer, &poly_standard);
                    poly_fft
                        .as_mut_tensor()
                        .fill_with_one(fft_buffer.as_tensor(), |a| *a);
                }

                glwe_out
                    .get_mut_body()
                    .as_mut_tensor()
                    .as_mut_slice()
                    .clone_from_slice(glwe_in.into_tensor().as_slice());
            });
    }
}

impl<Element, Cont, Scalar> AsRefTensor for FourierGgswSeededCiphertext<Cont, Scalar>
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

impl<Element, Cont, Scalar> AsMutTensor for FourierGgswSeededCiphertext<Cont, Scalar>
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

impl<Cont, Scalar> IntoTensor for FourierGgswSeededCiphertext<Cont, Scalar>
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
