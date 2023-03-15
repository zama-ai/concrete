use crate::implementation::{Container, ContainerMut, Split};

use super::ggsw_level_matrix::GgswLevelMatrix;
use super::{DecompParams, GlweParams};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GgswCiphertext<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub decomp_params: DecompParams,
}

impl<C: Container> GgswCiphertext<C> {
    pub fn data_len(glwe_params: GlweParams, decomposition_level_count: usize) -> usize {
        glwe_params.polynomial_size
            * (glwe_params.dimension + 1)
            * (glwe_params.dimension + 1)
            * decomposition_level_count
    }

    pub fn new(data: C, glwe_params: GlweParams, decomp_params: DecompParams) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(glwe_params, decomp_params.level));
        Self {
            data,
            glwe_params,
            decomp_params,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        decomp_params: DecompParams,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(data, Self::data_len(glwe_params, decomp_params.level));
        Self {
            data,
            glwe_params,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> GgswCiphertext<&[C::Item]> {
        GgswCiphertext {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
            decomp_params: self.decomp_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GgswCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GgswCiphertext {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            decomp_params: self.decomp_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_level_matrices_iter(self) -> impl DoubleEndedIterator<Item = GgswLevelMatrix<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.decomp_params.level)
            .enumerate()
            .map(move |(i, slice)| GgswLevelMatrix::new(slice, self.glwe_params, i + 1))
    }
}
