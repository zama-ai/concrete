use super::*;
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct PackingKeyswitchKey<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub input_dimension: usize,
    pub decomp_params: DecompParams,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlevCiphertext<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub ciphertext_count: usize,
}

impl<C: Container> PackingKeyswitchKey<C> {
    pub fn data_len(
        glwe_params: GlweParams,
        decomposition_level_count: usize,
        input_dimension: usize,
    ) -> usize {
        (input_dimension + 1)
            * decomposition_level_count
            * (glwe_params.dimension + 1)
            * glwe_params.polynomial_size
    }

    pub fn new(
        data: C,
        glwe_params: GlweParams,
        input_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(glwe_params, decomp_params.level, input_dimension),
        );

        Self {
            data,
            glwe_params,
            input_dimension,
            decomp_params,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        input_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(
                data,
                Self::data_len(glwe_params, decomp_params.level, input_dimension),
            ),
            glwe_params,
            input_dimension,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> PackingKeyswitchKey<&[C::Item]> {
        PackingKeyswitchKey {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
            input_dimension: self.input_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn as_mut_view(&mut self) -> PackingKeyswitchKey<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        PackingKeyswitchKey {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            input_dimension: self.input_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_glev_ciphertexts(self) -> impl DoubleEndedIterator<Item = GlevCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.input_dimension)
            .map(move |slice| {
                GlevCiphertext::new(slice, self.glwe_params, self.decomp_params.level)
            })
    }
}

impl<C: Container> GlevCiphertext<C> {
    pub fn data_len(glwe_params: GlweParams, ciphertext_count: usize) -> usize {
        glwe_params.polynomial_size * (glwe_params.dimension + 1) * ciphertext_count
    }

    pub fn new(data: C, glwe_params: GlweParams, ciphertext_count: usize) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(glwe_params, ciphertext_count));

        Self {
            data,
            glwe_params,
            ciphertext_count,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        ciphertext_count: usize,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(data, Self::data_len(glwe_params, ciphertext_count));
        Self {
            data,
            glwe_params,
            ciphertext_count,
        }
    }

    pub fn as_view(&self) -> GlevCiphertext<&[C::Item]> {
        GlevCiphertext {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
            ciphertext_count: self.ciphertext_count,
        }
    }

    pub fn as_mut_view(&mut self) -> GlevCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GlevCiphertext {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            ciphertext_count: self.ciphertext_count,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_ciphertext_iter(self) -> impl DoubleEndedIterator<Item = GlweCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.ciphertext_count)
            .map(move |slice| GlweCiphertext::new(slice, self.glwe_params))
    }
}
