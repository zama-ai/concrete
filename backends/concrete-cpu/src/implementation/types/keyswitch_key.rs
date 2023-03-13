use super::lev_ciphertext::LevCiphertext;
use super::*;
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct LweKeyswitchKey<C: Container> {
    pub data: C,
    pub output_dimension: usize,
    pub input_dimension: usize,
    pub decomp_params: DecompParams,
}

impl<C: Container> LweKeyswitchKey<C> {
    pub fn data_len(
        output_dimension: usize,
        decomposition_level_count: usize,
        input_dimension: usize,
    ) -> usize {
        input_dimension * decomposition_level_count * (output_dimension + 1)
    }

    pub fn new(
        data: C,
        output_dimension: usize,
        input_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(output_dimension, decomp_params.level, input_dimension),
        );

        Self {
            data,
            output_dimension,
            input_dimension,
            decomp_params,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        output_dimension: usize,
        input_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(
                data,
                Self::data_len(output_dimension, decomp_params.level, input_dimension),
            ),
            output_dimension,
            input_dimension,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> LweKeyswitchKey<&[C::Item]> {
        LweKeyswitchKey {
            data: self.data.as_ref(),
            output_dimension: self.output_dimension,
            input_dimension: self.input_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn as_mut_view(&mut self) -> LweKeyswitchKey<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        LweKeyswitchKey {
            data: self.data.as_mut(),
            output_dimension: self.output_dimension,
            input_dimension: self.input_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_lev_ciphertexts(self) -> impl DoubleEndedIterator<Item = LevCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.input_dimension)
            .map(move |slice| {
                LevCiphertext::new(slice, self.output_dimension, self.decomp_params.level)
            })
    }
}
