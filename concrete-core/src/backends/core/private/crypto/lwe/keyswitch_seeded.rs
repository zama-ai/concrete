#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

use concrete_commons::numeric::Numeric;

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, LweDimension, LweSize, Seed,
};

use crate::backends::core::private::crypto::encoding::{Plaintext, PlaintextList};
use crate::backends::core::private::crypto::secret::LweSecretKey;
use crate::backends::core::private::math::decomposition::{DecompositionLevel, DecompositionTerm};
use crate::backends::core::private::math::random::{RandomGenerable, Uniform};
use crate::backends::core::private::math::tensor::{
    ck_dim_div, tensor_traits, AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor,
};
use crate::backends::core::private::math::torus::UnsignedTorus;

use super::{LweKeyswitchKey, LweList, LweSeededCiphertext, LweSeededList};

/// A seeded Lwe Keyswithing key.
///
/// See [`LweKeyswitchKey`] for more details on keyswitching keys.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct LweSeededKeyswitchKey<Cont> {
    tensor: Tensor<Cont>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    lwe_size: LweSize,
    seed: Option<Seed>,
}

tensor_traits!(LweSeededKeyswitchKey);

impl<Scalar> LweSeededKeyswitchKey<Vec<Scalar>>
where
    Scalar: Copy,
{
    /// Allocates a seeded keyswitching key whose masks and bodies are all `value`.
    ///
    /// # Note
    ///
    /// This function does *not* generate a seeded keyswitch key, but merely allocates a container
    /// of the right size. See [`LweSeededKeyswitchKey::fill_with_seeded_keyswitch_key`] to fill
    /// the container with a proper seeded keyswitching key.
    pub fn allocate(
        value: Scalar,
        decomp_size: DecompositionLevelCount,
        decomp_base_log: DecompositionBaseLog,
        input_dimension: LweDimension,
        output_dimension: LweDimension,
    ) -> Self {
        Self {
            tensor: Tensor::from_container(vec![value; decomp_size.0 * input_dimension.0]),
            decomp_base_log,
            decomp_level_count: decomp_size,
            lwe_size: LweSize(output_dimension.0 + 1),
            seed: None,
        }
    }
}

impl<Cont> LweSeededKeyswitchKey<Cont> {
    /// Creates a seeded LWE key switching key from a container.
    ///
    /// # Notes
    ///
    /// This method does not create a seeded keyswitching key, but merely wrap the container in the
    /// proper type. It assumes that either the container already contains a proper seeded
    /// keyswitching key, or that [`LweSeededKeyswitchKey::fill_with_seeded_keyswitch_key`] will
    /// be called right after.
    pub fn from_container(
        cont: Cont,
        decomp_base_log: DecompositionBaseLog,
        decomp_size: DecompositionLevelCount,
        output_dimension: LweDimension,
        seed: Seed,
    ) -> Self
    where
        Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => decomp_size.0);
        LweSeededKeyswitchKey {
            tensor,
            decomp_base_log,
            decomp_level_count: decomp_size,
            lwe_size: LweSize(output_dimension.0 + 1),
            seed: Some(seed),
        }
    }

    /// Return the size of the output key.
    pub fn after_key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        self.lwe_size.to_lwe_dimension()
    }

    /// Returns the size of the ciphertexts encoding each level of the decomposition of each bits
    /// of the input key.
    pub fn lwe_size(&self) -> LweSize
    where
        Self: AsRefTensor,
    {
        self.lwe_size
    }

    /// Returns the size of the input key.
    pub fn before_key_size(&self) -> LweDimension
    where
        Self: AsRefTensor,
    {
        LweDimension(self.as_tensor().len() / self.decomp_level_count.0)
    }

    /// Returns the number of levels used for the decomposition of the input key bits.
    pub fn decomposition_levels_count(&self) -> DecompositionLevelCount
    where
        Self: AsRefTensor,
    {
        self.decomp_level_count
    }

    /// Returns the logarithm of the base used for the decomposition of the input key bits.
    ///
    /// Indeed, the basis used is always of the form $2^N$. This function returns $N$.
    pub fn decomposition_base_log(&self) -> DecompositionBaseLog
    where
        Self: AsRefTensor,
    {
        self.decomp_base_log
    }

    /// Fills the current seeded keyswitch key container with an actual seeded keyswitching key
    /// constructed from an input and an output key.
    pub fn fill_with_seeded_keyswitch_key<InKeyCont, OutKeyCont, Scalar>(
        &mut self,
        before_key: &LweSecretKey<BinaryKeyKind, InKeyCont>,
        after_key: &LweSecretKey<BinaryKeyKind, OutKeyCont>,
        noise_parameters: impl DispersionParameter,
        seed: Seed,
    ) where
        Self: AsMutTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, InKeyCont>: AsRefTensor<Element = Scalar>,
        LweSecretKey<BinaryKeyKind, OutKeyCont>: AsRefTensor<Element = Scalar>,
        Scalar: UnsignedTorus,
    {
        // We instantiate a buffer
        let mut messages = PlaintextList::from_container(vec![
            <Self as AsMutTensor>::Element::ZERO;
            self.decomp_level_count.0
        ]);

        // We retrieve decomposition arguments
        let decomp_level_count = self.decomp_level_count;
        let decomp_base_log = self.decomp_base_log;

        // loop over the before key blocks
        for (input_key_bit, keyswitch_key_block) in before_key
            .as_tensor()
            .iter()
            .zip(self.bit_decomp_iter_mut())
        {
            // We reset the buffer
            messages
                .as_mut_tensor()
                .fill_with_element(<Self as AsMutTensor>::Element::ZERO);

            // We fill the buffer with the powers of the key bits
            for (level, message) in (1..=decomp_level_count.0)
                .map(DecompositionLevel)
                .zip(messages.plaintext_iter_mut())
            {
                *message = Plaintext(
                    DecompositionTerm::new(level, decomp_base_log, *input_key_bit)
                        .to_recomposition_summand(),
                );
            }

            // We encrypt the buffer
            after_key.encrypt_seeded_lwe_list(
                &mut keyswitch_key_block.into_seeded_lwe_list(),
                &messages,
                noise_parameters,
                seed,
            );
        }
    }

    /// Iterates over borrowed `SeededLweKeyBitDecomposition` elements.
    ///
    /// One `SeededLweKeyBitDecomposition` being a set of seeded lwe ciphertexts, encrypting under
    /// the output key, the $l$ levels of the signed decomposition of a single bit of the input
    /// key.
    pub(crate) fn bit_decomp_iter(
        &self,
    ) -> impl Iterator<Item = SeededLweKeyBitDecomposition<&[<Self as AsRefTensor>::Element]>>
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        ck_dim_div!(self.as_tensor().len() => self.decomp_level_count.0);
        let level_count = self.decomp_level_count.0;
        let lwe_size = self.lwe_size;
        let seed = self.seed.unwrap();
        self.as_tensor()
            .subtensor_iter(level_count)
            .enumerate()
            .map(move |(i, sub)| {
                let seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift
                        + i * level_count * <Self as AsRefTensor>::Element::BITS / 8
                            * self.after_key_size().0,
                };
                SeededLweKeyBitDecomposition::from_container(sub.into_container(), lwe_size, seed)
            })
    }

    /// Iterates over mutably borrowed `SeededLweKeyBitDecomposition` elements.
    ///
    /// One `SeededLweKeyBitDecomposition` being a set of seeded lwe ciphertexts, encrypting under
    /// the output key, the $l$ levels of the signed decomposition of a single bit of the input
    /// key.
    pub(crate) fn bit_decomp_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = SeededLweKeyBitDecomposition<&mut [<Self as AsMutTensor>::Element]>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        ck_dim_div!(self.as_tensor().len() => self.decomp_level_count.0);
        let level_count = self.decomp_level_count.0;
        let lwe_size = self.lwe_size;
        let seed = self.seed.unwrap();
        let chunk_size =
            level_count * <Self as AsRefTensor>::Element::BITS / 8 * self.after_key_size().0;
        self.as_mut_tensor()
            .subtensor_iter_mut(level_count)
            .enumerate()
            .map(move |(i, sub)| {
                let seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift + i * chunk_size,
                };
                SeededLweKeyBitDecomposition::from_container(sub.into_container(), lwe_size, seed)
            })
    }

    pub(super) fn get_seed(&self) -> Option<Seed> {
        self.seed
    }

    pub(super) fn get_mut_seed(&mut self) -> &mut Option<Seed> {
        &mut self.seed
    }

    pub fn expand_into<OutCont, Scalar>(self, output: &mut LweKeyswitchKey<OutCont>)
    where
        LweKeyswitchKey<OutCont>: AsMutTensor<Element = Scalar>,
        Self: AsRefTensor<Element = Scalar>,
        Scalar: Copy + RandomGenerable<Uniform> + Numeric,
    {
        for (mut output_tensor, keyswitch_key_block) in output
            .as_mut_tensor()
            .subtensor_iter_mut(self.decomp_level_count.0)
            .zip(self.bit_decomp_iter())
        {
            let mut lwe_list = LweList::from_container(output_tensor.as_mut_slice(), self.lwe_size);
            keyswitch_key_block
                .into_seeded_lwe_list()
                .expand_into(&mut lwe_list);
        }
    }
}

/// The encryption of a single bit of the output key.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
pub(crate) struct SeededLweKeyBitDecomposition<Cont> {
    pub(super) tensor: Tensor<Cont>,
    pub(super) lwe_size: LweSize,
    pub(super) seed: Seed,
}

tensor_traits!(SeededLweKeyBitDecomposition);

impl<Cont> SeededLweKeyBitDecomposition<Cont> {
    /// Creates a key bit decomposition from a container.
    ///
    /// # Notes
    ///
    /// This method does not decompose a key bit in a basis, but merely wraps a container in the
    /// right structure. See [`LweSeededKeyswitchKey::bit_decomp_iter`] for an iterator that returns
    /// key bit decompositions.
    pub fn from_container(cont: Cont, lwe_size: LweSize, seed: Seed) -> Self {
        SeededLweKeyBitDecomposition {
            tensor: Tensor::from_container(cont),
            lwe_size,
            seed,
        }
    }

    /// Returns the size of the lwe ciphertexts encoding each level of the key bit decomposition.
    pub fn lwe_size(&self) -> LweSize {
        self.lwe_size
    }

    /// Returns the number of ciphertexts in the decomposition.
    ///
    /// Note that this is actually equals to the number of levels in the decomposition.
    pub fn count(&self) -> CiphertextCount
    where
        Self: AsRefTensor,
    {
        CiphertextCount(self.as_tensor().len())
    }

    /// Returns an iterator over borrowed `LweSeededCiphertext`.
    pub fn seeded_ciphertext_iter(
        &self,
    ) -> impl Iterator<Item = LweSeededCiphertext<&<Self as AsRefTensor>::Element>>
    where
        Self: AsRefTensor,
        <Self as AsRefTensor>::Element: Numeric,
    {
        let lwe_dimension = self.lwe_size.to_lwe_dimension();
        let seed = self.seed;
        let byte_size = <Self as AsRefTensor>::Element::BITS / 8;

        self.as_tensor().iter().enumerate().map(move |(i, sub)| {
            let ctx_seed = Seed {
                seed: seed.seed,
                shift: seed.shift + i * lwe_dimension.0 * byte_size,
            };
            LweSeededCiphertext::from_scalar(sub, lwe_dimension, ctx_seed)
        })
    }

    /// Returns an iterator over mutably borrowed `LweSeededCiphertext`.
    pub fn seeded_ciphertext_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = LweSeededCiphertext<&mut <Self as AsMutTensor>::Element>>
    where
        Self: AsMutTensor,
        <Self as AsMutTensor>::Element: Numeric,
    {
        let lwe_dimension = self.lwe_size.to_lwe_dimension();
        let seed = self.seed;
        let byte_size = <Self as AsRefTensor>::Element::BITS / 8;

        self.as_mut_tensor()
            .iter_mut()
            .enumerate()
            .map(move |(i, sub)| {
                let ctx_seed = Seed {
                    seed: seed.seed,
                    shift: seed.shift + i * lwe_dimension.0 * byte_size,
                };
                LweSeededCiphertext::from_scalar(sub, lwe_dimension, ctx_seed)
            })
    }

    /// Consumes the current key bit decomposition and returns a seeded lwe list.
    ///
    /// Note that this operation is super cheap, as it merely rewraps the current container in a
    /// seeded lwe list structure.
    pub fn into_seeded_lwe_list(self) -> LweSeededList<Cont>
    where
        Cont: AsRefSlice,
    {
        LweSeededList::from_container(
            self.tensor.into_container(),
            self.lwe_size.to_lwe_dimension(),
            self.seed,
        )
    }
}
