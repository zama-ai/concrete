use super::{GlweCiphertext, GlweParams};
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GgswLevelMatrix<C: Container> {
    pub data: C,
    pub polynomial_size: usize,
    pub in_glwe_dimension: usize,
    pub out_glwe_dimension: usize,
    pub decomposition_level: usize,
}

impl<C: Container> GgswLevelMatrix<C> {
    pub fn out_glwe_params(&self) -> GlweParams {
        GlweParams {
            dimension: self.out_glwe_dimension,
            polynomial_size: self.polynomial_size,
        }
    }

    pub fn in_glwe_params(&self) -> GlweParams {
        GlweParams {
            dimension: self.in_glwe_dimension,
            polynomial_size: self.polynomial_size,
        }
    }

    pub fn data_len(
        polynomial_size: usize,
        in_glwe_dimension: usize,
        out_glwe_dimension: usize,
    ) -> usize {
        polynomial_size * (in_glwe_dimension + 1) * (out_glwe_dimension + 1)
    }

    pub fn new(
        data: C,
        polynomial_size: usize,
        in_glwe_dimension: usize,
        out_glwe_dimension: usize,
        decomposition_level: usize,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(polynomial_size, in_glwe_dimension, out_glwe_dimension,)
        );
        Self {
            data,
            polynomial_size,
            in_glwe_dimension,
            out_glwe_dimension,
            decomposition_level,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        polynomial_size: usize,
        in_glwe_dimension: usize,
        out_glwe_dimension: usize,
        decomposition_level: usize,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(
            data,
            Self::data_len(polynomial_size, in_glwe_dimension, out_glwe_dimension),
        );
        Self {
            data,
            polynomial_size,
            in_glwe_dimension,
            out_glwe_dimension,
            decomposition_level,
        }
    }

    pub fn as_view(&self) -> GgswLevelMatrix<&[C::Item]> {
        GgswLevelMatrix {
            data: self.data.as_ref(),
            polynomial_size: self.polynomial_size,
            in_glwe_dimension: self.in_glwe_dimension,
            out_glwe_dimension: self.out_glwe_dimension,
            decomposition_level: self.decomposition_level,
        }
    }

    pub fn as_mut_view(&mut self) -> GgswLevelMatrix<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GgswLevelMatrix {
            data: self.data.as_mut(),
            polynomial_size: self.polynomial_size,
            in_glwe_dimension: self.in_glwe_dimension,
            out_glwe_dimension: self.out_glwe_dimension,
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
        let out_glwe_params = self.out_glwe_params();

        self.data
            .split_into(self.in_glwe_dimension + 1)
            .map(move |slice| GlweCiphertext::new(slice, out_glwe_params))
    }
}
