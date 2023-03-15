use super::{GlweCiphertext, GlweParams};
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GgswLevelMatrix<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub decomposition_level: usize,
}

impl<C: Container> GgswLevelMatrix<C> {
    pub fn data_len(glwe_params: GlweParams) -> usize {
        glwe_params.polynomial_size * (glwe_params.dimension + 1) * (glwe_params.dimension + 1)
    }

    pub fn new(data: C, glwe_params: GlweParams, decomposition_level: usize) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(glwe_params));
        Self {
            data,
            glwe_params,
            decomposition_level,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        decomposition_level: usize,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(data, Self::data_len(glwe_params));
        Self {
            data,
            glwe_params,
            decomposition_level,
        }
    }

    pub fn as_view(&self) -> GgswLevelMatrix<&[C::Item]> {
        GgswLevelMatrix {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
            decomposition_level: self.decomposition_level,
        }
    }

    pub fn as_mut_view(&mut self) -> GgswLevelMatrix<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GgswLevelMatrix {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            decomposition_level: self.decomposition_level,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_rows_iter(self) -> impl DoubleEndedIterator<Item = GlweCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.glwe_params.dimension + 1)
            .map(move |slice| GlweCiphertext::new(slice, self.glwe_params))
    }
}
