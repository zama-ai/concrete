use concrete_commons::numeric::Numeric;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, LweSize};
use concrete_csprng::RandomGenerator;
#[cfg(feature = "multithread")]
use rayon::{iter::IndexedParallelIterator, prelude::*};
use serde::{Deserialize, Serialize};

use crate::backends::core::private::crypto::lwe::LweSeededList;
use crate::backends::core::private::crypto::secret::generators::EncryptionRandomGenerator;
use crate::backends::core::private::math::decomposition::DecompositionLevel;
use crate::backends::core::private::math::random::{RandomGenerable, Uniform};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};

use super::{GswCiphertext, GswSeededLevelMatrix};

/// A GSW seeded ciphertext.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct GswSeededCiphertext<Cont> {
    tensor: Tensor<Cont>,
    lwe_size: LweSize,
    decomp_base_log: DecompositionBaseLog,
    seed: u128,
    shift: usize,
}

tensor_traits!(GswSeededCiphertext);

impl<Scalar> GswSeededCiphertext<Vec<Scalar>> {
    /// Allocates a new GSW seeded ciphertext whose coefficients are all `value`.
    pub fn allocate(
        value: Scalar,
        lwe_size: LweSize,
        decomp_level: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
    ) -> Self
    where
        Scalar: Numeric,
    {
        GswSeededCiphertext {
            tensor: Tensor::from_container(vec![value; decomp_level.0 * lwe_size.0]),
            lwe_size,
            decomp_base_log,
            seed: RandomGenerator::generate_u128(),
            shift: 0,
        }
    }
}

impl<Cont> GswSeededCiphertext<Cont> {
    /// Creates a gsw ciphertext from an existing container.
    pub fn from_container(
        cont: Cont,
        lwe_size: LweSize,
        decomp_base_log: DecompositionBaseLog,
        seed: u128,
        shift: usize,
    ) -> Self
    where
        Cont: AsRefSlice,
        <Cont as AsRefSlice>::Element: Numeric,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => lwe_size.0);
        GswSeededCiphertext {
            tensor,
            lwe_size,
            decomp_base_log,
            seed,
            shift,
        }
    }

    /// Returns the size of the lwe ciphertexts composing the gsw seeded ciphertext.
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    pub fn get_seed(&self) -> u128 {
        self.seed
    }

    /// Returns the number of decomposition levels used in the seeded ciphertext.
    pub fn decomposition_level_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        ck_dim_div!(self.as_tensor().len() =>
            self.lwe_size.0
        );
        DecompositionLevelCount(self.as_tensor().len() / self.lwe_size.0)
    }

    /// Returns a borrowed list composed of all the LWE seeded ciphertexts composing current ciphertext.
    pub fn as_lwe_list<Scalar>(&self) -> LweSeededList<&[Scalar]>
    where
        Self: AsRefTensor<Element = Scalar>,
    {
        LweSeededList::from_container(
            self.as_tensor().as_slice(),
            self.lwe_size.to_lwe_dimension(),
            self.seed,
            self.shift,
        )
    }

    /// Returns a mutably borrowed `LweSeededList` composed of all the LWE seeded ciphertexts composing
    /// current ciphertext.
    pub fn as_mut_lwe_list<Scalar>(&mut self) -> LweSeededList<&mut [Scalar]>
    where
        Self: AsMutTensor<Element = Scalar>,
    {
        let lwe_dimension = self.lwe_size.to_lwe_dimension();
        let seed = self.seed;
        let shift = self.shift;
        LweSeededList::from_container(
            self.as_mut_tensor().as_mut_slice(),
            lwe_dimension,
            seed,
            shift,
        )
    }

    /// Returns the logarithm of the base used for the gadget decomposition.
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    /// Returns an iterator over borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    pub fn level_matrix_iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = GswSeededLevelMatrix<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
    {
        let lwe_size = self.lwe_size;
        self.as_tensor()
            .subtensor_iter(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    self.seed,
                    index * lwe_size.0,
                )
            })
    }

    /// Returns an iterator over mutably borrowed seeded level matrices.
    ///
    /// # Note
    ///
    /// This iterator iterates over the levels from the lower to the higher level in the usual
    /// order. To iterate in the reverse order, you can use `rev()` on the iterator.
    pub fn level_matrix_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = GswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
    {
        let lwe_size = self.lwe_size;
        let seed = self.seed;
        self.as_mut_tensor()
            .subtensor_iter_mut(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    seed,
                    index * lwe_size.0,
                )
            })
    }

    /// Returns a parallel iterator over mutably borrowed seeded level matrices.
    ///
    /// # Notes
    /// This iterator is hidden behind the "multithread" feature gate.
    #[cfg(feature = "multithread")]
    pub fn par_level_matrix_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = GswSeededLevelMatrix<&mut [<Self as AsRefTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Sync + Send,
    {
        let lwe_size = self.lwe_size;
        let seed = self.seed;
        self.as_mut_tensor()
            .par_subtensor_iter_mut(lwe_size.0)
            .enumerate()
            .map(move |(index, tensor)| {
                GswSeededLevelMatrix::from_container(
                    tensor.into_container(),
                    lwe_size,
                    DecompositionLevel(index + 1),
                    seed,
                    index * lwe_size.0,
                )
            })
    }

    pub fn expand_into<OutCont, Scalar>(self, output: &mut GswCiphertext<OutCont, Scalar>)
    where
        GswCiphertext<OutCont, Scalar>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: Copy + RandomGenerable<Uniform>,
    {
        let mut generator = EncryptionRandomGenerator::new(Some(self.seed));
        for (mut lwe_out, body_in) in output
            .as_mut_lwe_list()
            .ciphertext_iter_mut()
            .zip(self.as_lwe_list().ciphertext_iter())
        {
            let (mut body, mut mask) = lwe_out.get_mut_body_and_mask();
            generator.fill_tensor_with_random_mask(mask.as_mut_tensor());
            body.0 = body_in.get_body().0;
        }
    }
}
