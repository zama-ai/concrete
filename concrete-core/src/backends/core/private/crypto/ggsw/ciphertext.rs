#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
};

use crate::backends::core::private::crypto::glwe::GlweList;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::GgswLevelMatrix;

/// A GGSW ciphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GgswCiphertext<Cont> {
    tensor: Tensor<Cont>,
    poly_size: PolynomialSize,
    rlwe_size: GlweSize,
    decomp_base_log: DecompositionBaseLog,
}

tensor_traits!(GgswCiphertext);

impl<Scalar> GgswCiphertext<Vec<Scalar>> {
    /// Allocates a new GGSW ciphertext whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn allocate(
        value: Scalar,
        poly_size: PolynomialSize,
        rlwe_size: GlweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Copy,
    {
        GgswCiphertext {
            tensor: Tensor::from_container(vec![
                value;
                decomp_level.0
                    * rlwe_size.0
                    * rlwe_size.0
                    * poly_size.0
            ]),
            poly_size,
            rlwe_size,
            decomp_base_log,
        }
    }
}

impl<Cont> GgswCiphertext<Cont> {
    /// Creates an Rgsw ciphertext from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::from_container(
    ///     vec![9 as u8; 7 * 7 * 10 * 3],
    ///     GlweSize(7),
    ///     PolynomialSize(10),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// assert_eq!(ggsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(ggsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn from_container(
        cont: Cont,
        rlwe_size: GlweSize,
        poly_size: PolynomialSize,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => rlwe_size.0, poly_size.0, rlwe_size.0 * rlwe_size.0);
        GgswCiphertext {
            tensor,
            poly_size,
            rlwe_size,
            decomp_base_log,
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
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(ggsw.glwe_size(), GlweSize(7));
    /// ```
    pub fn glwe_size(&self) -> GlweSize {
        self.rlwe_size
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
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
            self.rlwe_size.0,
            self.poly_size.0,
            self.rlwe_size.0 * self.rlwe_size.0
        );
        DecompositionLevelCount(
            self.as_tensor().len() / (self.rlwe_size.0 * self.rlwe_size.0 * self.poly_size.0),
        )
    }

    /// Returns the size of the polynomials used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
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
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let list = ggsw.as_glwe_list();
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_glwe_list<Scalar>(&self) -> GlweList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        GlweList::from_container(
            self.as_tensor().as_slice(),
            self.rlwe_size.to_glwe_dimension(),
            self.poly_size,
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
    ///     PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(10),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let mut list = ggsw.as_mut_glwe_list();
    /// list.as_mut_tensor().fill_with_element(0);
    /// assert_eq!(list.glwe_dimension(), GlweDimension(6));
    /// assert_eq!(list.ciphertext_count(), CiphertextCount(3 * 7));
    /// ggsw.as_tensor().iter().for_each(|a| assert_eq!(*a, 0));
    /// ```
    pub fn as_mut_glwe_list<Scalar>(&mut self) -> GlweList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let dimension = self.rlwe_size.to_glwe_dimension();
        let size = self.poly_size;
        GlweList::from_container(self.as_mut_tensor().as_mut_slice(), dimension, size)
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
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
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// let ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for level_matrix in ggsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     assert_eq!(level_matrix.polynomial_size(), PolynomialSize(9));
    ///     for rlwe in level_matrix.row_iter() {
    ///         assert_eq!(rlwe.glwe_size(), GlweSize(7));
    ///         assert_eq!(rlwe.polynomial_size(), PolynomialSize(9));
    ///     }
    /// }
    /// assert_eq!(ggsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GgswLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
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
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for mut level_matrix in ggsw.level_matrix_iter_mut() {
    ///     for mut rlwe in level_matrix.row_iter_mut() {
    ///         rlwe.as_mut_tensor().fill_with_element(9);
    ///     }
    /// }
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GgswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
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
    ///     DecompositionBaseLog, DecompositionLevelCount, GlweSize, PolynomialSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::ggsw::GgswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    ///
    /// let mut ggsw = GgswCiphertext::allocate(
    ///     9 as u8,
    ///     PolynomialSize(9),
    ///     GlweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// ggsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut rlwe in level_matrix.row_iter_mut() {
    ///             rlwe.as_mut_tensor().fill_with_element(9);
    ///         }
    ///     });
    /// assert!(ggsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(ggsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GgswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
    {
        let chunks_size = self.poly_size.0 * self.rlwe_size.0 * self.rlwe_size.0;
        let poly_size = self.poly_size;
        let rlwe_size = self.rlwe_size;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GgswLevelMatrix::from_container(
                    tensor.into_container(),
                    poly_size,
                    rlwe_size,
                    DecompositionLevel(index + 1),
                )
            })
    }
}
