use std::cell::RefCell;

use crate::backends::core::private::crypto::lwe::{LweCiphertext, LweList};
use crate::backends::core::private::math::decomposition::{DecompositionLevel, SignedDecomposer};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, ck_dim_eq, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;
use crate::backends::core::private::utils::zip;

use super::GswLevelMatrix;

use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A GSW ciphertext.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct GswCiphertext<Cont, Scalar> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    decomp_base_log: DecompositionBaseLog,
    rounded_buffer: RefCell<LweCiphertext<Vec<Scalar>>>,
}

impl<Scalar> GswCiphertext<Vec<Scalar>, Scalar> {
    /// Allocates a new GSW ciphertext whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn allocate(
        value: Scalar,
        lwe_size: LweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Numeric,
    {
        GswCiphertext {
            tensor: Tensor::from_container(vec![value; decomp_level.0 * lwe_size.0 * lwe_size.0]),
            lwe_size,
            decomp_base_log,
            rounded_buffer: RefCell::new(LweCiphertext::allocate(Scalar::ZERO, lwe_size)),
        }
    }
}

impl<Cont, Scalar> GswCiphertext<Cont, Scalar> {
    /// Creates a gsw ciphertext from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::from_container(
    ///     vec![9 as u8; 7 * 7 * 3],
    ///     LweSize(7),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
    /// ```
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Cont: AsRefSlice<Element = Scalar>,
        Scalar: Numeric,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => lwe_size.0,lwe_size.0 * lwe_size.0);
        GswCiphertext {
            tensor,
            lwe_size,
            decomp_base_log,
            rounded_buffer: RefCell::new(LweCiphertext::allocate(Scalar::ZERO, lwe_size)),
        }
    }

    /// Returns the size of the lwe ciphertexts composing the gsw ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.lwe_size(), LweSize(7));
    /// ```
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the number of decomposition levels used in the ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.decomposition_level_count(), DecompositionLevelCount(3));
    /// ```
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.lwe_size.0,
            self.lwe_size.0 * self.lwe_size.0
        );
        DecompositionLevelCount(self.as_tensor().len() / (self.lwe_size.0 * self.lwe_size.0))
    }

    /// Returns a borrowed list composed of all the LWE ciphertexts composing current ciphertext.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let list = gsw.as_lwe_list();
    /// assert_eq!(list.lwe_size(), LweSize(7));
    /// assert_eq!(list.count(), CiphertextCount(3 * 7));
    /// ```
    pub fn as_lwe_list(&self) -> LweList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        LweList::from_container(self.as_tensor().as_slice(), self.lwe_size)
    }

    /// Returns a mutably borrowed `LweList` composed of all the LWE ciphertexts composing
    /// current ciphertext.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{
    ///     CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// let mut list = gsw.as_mut_lwe_list();
    /// list.as_mut_tensor().fill_with_element(0);
    /// assert_eq!(list.lwe_size(), LweSize(7));
    /// assert_eq!(list.count(), CiphertextCount(3 * 7));
    /// gsw.as_tensor().iter().for_each(|a| assert_eq!(*a, 0));
    /// ```
    pub fn as_mut_lwe_list(&mut self) -> LweList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let lwe_size = self.lwe_size;
        LweList::from_container(self.as_mut_tensor().as_mut_slice(), lwe_size)
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// assert_eq!(gsw.decomposition_base_log(), DecompositionBaseLog(4));
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
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// let gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for level_matrix in gsw.level_matrix_iter() {
    ///     assert_eq!(level_matrix.row_iter().count(), 7);
    ///     for lwe in level_matrix.row_iter() {
    ///         assert_eq!(lwe.lwe_size(), LweSize(7));
    ///     }
    /// }
    /// assert_eq!(gsw.level_matrix_iter().count(), 3);
    /// ```
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GswLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let chunks_size = self.lwe_size.0 * self.lwe_size.0;
        let lwe_size = self.lwe_size;
        self.as_tensor()
            .subtensor_iter(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GswLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
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
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// let mut gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// for mut level_matrix in gsw.level_matrix_iter_mut() {
    ///     for mut lwe in level_matrix.row_iter_mut() {
    ///         lwe.as_mut_tensor().fill_with_element(9);
    ///     }
    /// }
    /// assert!(gsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(gsw.level_matrix_iter_mut().count(), 3);
    /// ```
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let chunks_size = self.lwe_size.0 * self.lwe_size.0;
        let lwe_size = self.lwe_size;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GswLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
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
    /// use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// use concrete_core::backends::core::private::math::tensor::{AsMutTensor, AsRefTensor};
    /// use rayon::iter::ParallelIterator;
    /// let mut gsw = GswCiphertext::allocate(
    ///     9 as u8,
    ///     LweSize(7),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(4),
    /// );
    /// gsw.par_level_matrix_iter_mut()
    ///     .for_each(|mut level_matrix| {
    ///         for mut lwe in level_matrix.row_iter_mut() {
    ///             lwe.as_mut_tensor().fill_with_element(9);
    ///         }
    ///     });
    /// assert!(gsw.as_tensor().iter().all(|a| *a == 9));
    /// assert_eq!(gsw.level_matrix_iter_mut().count(), 3);
    /// ```
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GswLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
    {
        let chunks_size = self.lwe_size.0 * self.lwe_size.0;
        let lwe_size = self.lwe_size;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(chunks_size)
            .enumerate()
            .map(move |(index, tensor)| {
                GswLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                )
            })
    }

    /// Computes the external product and adds it to the output
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    ///
    /// use concrete_commons::parameters::LweDimension;
    /// use concrete_commons::parameters::LweSize;
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_commons::parameters::DecompositionLevelCount;
    /// use concrete_commons::parameters::DecompositionBaseLog;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    ///
    /// let lwe_sk = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);

    /// let mut gsw = GswCiphertext::allocate(
    ///     0 as u32,
    ///     LweSize(257),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(7),
    /// );
    /// let std_dev = LogStandardDev(-20.);
    /// lwe_sk.encrypt_constant_gsw(
    ///     &mut gsw,
    ///     &Plaintext(1 as u32),
    ///     std_dev,
    ///     &mut encryption_generator,
    /// );
    ///
    /// let mut ciphertext = LweCiphertext::allocate(0 as u32, LweSize(257));
    /// let mut res = LweCiphertext::allocate(0 as u32, LweSize(257));
    ///
    /// lwe_sk.encrypt_lwe(
    ///     &mut ciphertext,
    ///     &Plaintext(0 as u32),
    ///     std_dev,
    ///     &mut encryption_generator,
    /// );
    ///
    /// gsw.external_product(&mut res, &ciphertext);
    /// ```
    pub fn external_product<C1, C2>(&self, output: &mut LweCiphertext<C1>, lwe: &LweCiphertext<C2>)
    where
        Self: AsRefTensor<Element = Scalar>,
        LweCiphertext<C1>: AsMutTensor<Element = Scalar>,
        LweCiphertext<C2>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We check that the lwe sizes match
        ck_dim_eq!(
            self.lwe_size =>
            lwe.lwe_size(),
            output.lwe_size()
        );

        // We mutably borrow a standard domain buffer to store the rounded input.
        let rounded_input_lwe = &mut *self.rounded_buffer.borrow_mut();

        // We round the input mask and body
        let decomposer =
            SignedDecomposer::new(self.decomp_base_log, self.decomposition_level_count());
        decomposer.fill_tensor_with_closest_representable(rounded_input_lwe, lwe);

        let mut decomposition = decomposer.decompose_tensor(rounded_input_lwe);
        // We loop through the levels (we reverse to match the order of the decomposition iterator.)
        for gsw_decomp_matrix in self.level_matrix_iter().rev() {
            // We retrieve the decomposition of this level.
            let lwe_decomp_term = decomposition.next_term().unwrap();
            debug_assert_eq!(
                gsw_decomp_matrix.decomposition_level(),
                lwe_decomp_term.level()
            );
            // For each levels we have to add the result of the vector-matrix product between the
            // decomposition of the lwe, and the gsw level matrix to the output. To do so, we
            // iteratively add to the output, the product between every lines of the matrix, and
            // the corresponding scalar in the lwe decomposition:
            //
            //                gsw_mat                         gsw_mat
            //   lwe_dec    | - - - - | <        lwe_dec    | - - - - |
            //  | - - - | x | - - - - |         | - - - | x | - - - - | <
            //    ^         | - - - - |             ^       | - - - - |
            //
            //        t = 1                           t = 2                     ...
            let iterator = zip!(
                gsw_decomp_matrix.row_iter(),
                lwe_decomp_term.as_tensor().iter()
            );

            //---------------------------------------------------------------- VECTOR-MATRIX PRODUCT
            for (gsw_row, lwe_coeff) in iterator {
                // We loop through the coefficients of the output, and add the
                // corresponding product of scalars.
                output.as_mut_tensor().update_with_one(
                    gsw_row.as_tensor(),
                    |output_coeff, gsw_coeff| {
                        *output_coeff =
                            output_coeff.wrapping_add(gsw_coeff.wrapping_mul(*lwe_coeff))
                    },
                );
            }
        }
    }

    /// Computes the CMux between ct0 and ct1 and writes the result in ouptut
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::dispersion::LogStandardDev;
    ///
    /// use concrete_commons::parameters::{
    ///     DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize,
    /// };
    /// use concrete_core::backends::core::private::crypto::encoding::Plaintext;
    /// use concrete_core::backends::core::private::crypto::gsw::GswCiphertext;
    /// use concrete_core::backends::core::private::crypto::lwe::LweCiphertext;
    /// use concrete_core::backends::core::private::crypto::secret::generators::{
    ///     EncryptionRandomGenerator, SecretRandomGenerator,
    /// };
    /// use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
    ///
    /// let mut secret_generator = SecretRandomGenerator::new(None);
    /// let mut encryption_generator = EncryptionRandomGenerator::new(None);
    ///
    /// let lwe_sk = LweSecretKey::generate_binary(LweDimension(256), &mut secret_generator);
    ///
    /// let mut gsw = GswCiphertext::allocate(
    ///     0 as u32,
    ///     LweSize(257),
    ///     DecompositionLevelCount(3),
    ///     DecompositionBaseLog(7),
    /// );
    /// let std_dev = LogStandardDev(-20.);
    /// lwe_sk.encrypt_constant_gsw(
    ///     &mut gsw,
    ///     &Plaintext(1 as u32),
    ///     std_dev,
    ///     &mut encryption_generator,
    /// );
    ///
    /// let mut ciphertext0 = LweCiphertext::allocate(0 as u32, LweSize(257));
    /// let mut ciphertext1 = LweCiphertext::allocate(0 as u32, LweSize(257));
    /// let mut out = LweCiphertext::allocate(0 as u32, LweSize(257));
    ///
    /// lwe_sk.encrypt_lwe(
    ///     &mut ciphertext0,
    ///     &Plaintext(0 as u32),
    ///     std_dev,
    ///     &mut encryption_generator,
    /// );
    /// lwe_sk.encrypt_lwe(
    ///     &mut ciphertext1,
    ///     &Plaintext(1 as u32),
    ///     std_dev,
    ///     &mut encryption_generator,
    /// );
    ///
    /// gsw.cmux(&mut out, &ciphertext0, &ciphertext1);
    /// ```
    pub fn cmux<C0, C1, COut>(
        &self,
        output: &mut LweCiphertext<COut>,
        ct0: &LweCiphertext<C0>,
        ct1: &LweCiphertext<C1>,
    ) where
        LweCiphertext<C0>: AsRefTensor<Element = Scalar>,
        LweCiphertext<C1>: AsRefTensor<Element = Scalar>,
        LweCiphertext<Vec<Scalar>>: AsMutTensor<Element = Scalar>,
        LweCiphertext<COut>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        let mut buffer = LweCiphertext::allocate(Scalar::ZERO, ct1.lwe_size());
        buffer
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct1.as_tensor().as_slice());
        output
            .as_mut_tensor()
            .as_mut_slice()
            .clone_from_slice(ct0.as_tensor().as_slice());
        buffer
            .as_mut_tensor()
            .update_with_wrapping_sub(ct0.as_tensor());
        self.external_product(output, &buffer);
    }
}

impl<Element, Cont, Scalar> AsRefTensor for GswCiphertext<Cont, Scalar>
where
    Cont: AsRefSlice<Element = Element>,
    Scalar: Numeric,
{
    type Element = Element;
    type Container = Cont;
    fn as_tensor(&self) -> &Tensor<Self::Container> {
        &self.tensor
    }
}

impl<Element, Cont, Scalar> AsMutTensor for GswCiphertext<Cont, Scalar>
where
    Cont: AsMutSlice<Element = Element>,
    Scalar: Numeric,
{
    type Element = Element;
    type Container = Cont;
    fn as_mut_tensor(&mut self) -> &mut Tensor<<Self as AsMutTensor>::Container> {
        &mut self.tensor
    }
}

impl<Cont, Scalar> IntoTensor for GswCiphertext<Cont, Scalar>
where
    Cont: AsRefSlice,
    Scalar: Numeric,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}
