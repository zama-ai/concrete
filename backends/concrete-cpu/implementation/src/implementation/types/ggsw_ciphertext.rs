use crate::implementation::{Container, ContainerMut, Split};

use super::ggsw_level_matrix::GgswLevelMatrix;
use super::{DecompParams, GlweParams};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GgswCiphertext<C: Container> {
    pub data: C,
    pub polynomial_size: usize,
    pub in_glwe_dimension: usize,
    pub out_glwe_dimension: usize,
    pub decomp_params: DecompParams,
}

impl<C: Container> GgswCiphertext<C> {
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
        decomposition_level_count: usize,
    ) -> usize {
        decomposition_level_count
            * (in_glwe_dimension + 1)
            * (out_glwe_dimension + 1)
            * polynomial_size
    }

    pub fn new(
        data: C,
        polynomial_size: usize,
        in_glwe_dimension: usize,
        out_glwe_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(
                polynomial_size,
                in_glwe_dimension,
                out_glwe_dimension,
                decomp_params.level
            )
        );
        Self {
            data,
            polynomial_size,
            in_glwe_dimension,
            out_glwe_dimension,
            decomp_params,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        polynomial_size: usize,
        in_glwe_dimension: usize,
        out_glwe_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(
            data,
            Self::data_len(
                polynomial_size,
                in_glwe_dimension,
                out_glwe_dimension,
                decomp_params.level,
            ),
        );
        Self {
            data,
            polynomial_size,
            in_glwe_dimension,
            out_glwe_dimension,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> GgswCiphertext<&[C::Item]> {
        GgswCiphertext {
            data: self.data.as_ref(),
            polynomial_size: self.polynomial_size,
            in_glwe_dimension: self.in_glwe_dimension,
            out_glwe_dimension: self.out_glwe_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GgswCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GgswCiphertext {
            data: self.data.as_mut(),
            polynomial_size: self.polynomial_size,
            in_glwe_dimension: self.in_glwe_dimension,
            out_glwe_dimension: self.out_glwe_dimension,
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
            .map(move |(i, slice)| {
                GgswLevelMatrix::new(
                    slice,
                    self.polynomial_size,
                    self.in_glwe_dimension,
                    self.out_glwe_dimension,
                    i + 1,
                )
            })
    }
}
