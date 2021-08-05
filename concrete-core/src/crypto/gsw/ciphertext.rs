use std::cell::RefCell;

use crate::crypto::lwe::{LweCiphertext, LweList};
use crate::math::decomposition::{DecompositionLevel, SignedDecomposer};
use crate::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, IntoTensor, Tensor};
use crate::math::torus::UnsignedTorus;
use crate::{ck_dim_div, ck_dim_eq, zip, zip_args};

use super::GswLevelMatrix;

use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};

/// A GSW ciphertext.
pub struct GswCiphertext<Cont, Scalar> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    decomp_base_log: DecompositionBaseLog,
    rounded_buffer: RefCell<LweCiphertext<Vec<Scalar>>>,
}

impl<Scalar> GswCiphertext<Vec<Scalar>, Scalar> {
    /// Allocates a new GSW ciphertext whose coefficients are all `value`.
    pub fn allocate(
        value: Scalar,
        lwe_size: LweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: UnsignedTorus,
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
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Cont: AsRefSlice,
        Scalar: UnsignedTorus,
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
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the number of decomposition levels used in the ciphertext.
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
    pub fn as_lwe_list(&self) -> LweList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        LweList::from_container(self.as_tensor().as_slice(), self.lwe_size)
    }

    /// Returns a mutably borrowed `LweList` composed of all the LWE ciphertexts composing
    /// current ciphertext.
    pub fn as_mut_lwe_list(&mut self) -> LweList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let lwe_size = self.lwe_size;
        LweList::from_container(self.as_mut_tensor().as_mut_slice(), lwe_size)
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns an iterator over borrowed level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
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
                let iterator = zip!(
                    gsw_row.as_tensor().iter(),
                    output.as_mut_tensor().iter_mut()
                );
                for zip_args!(gsw_coeff, output_coeff) in iterator {
                    *output_coeff += *gsw_coeff * *lwe_coeff;
                }
            }
        }
    }

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
    Scalar: UnsignedTorus,
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
    Scalar: UnsignedTorus,
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
    Scalar: UnsignedTorus,
{
    type Element = <Cont as AsRefSlice>::Element;
    type Container = Cont;
    fn into_tensor(self) -> Tensor<Self::Container> {
        self.tensor
    }
}
