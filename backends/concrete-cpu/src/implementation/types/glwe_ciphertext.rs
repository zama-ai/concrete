use crate::implementation::{zip_eq, Container, ContainerMut, Split};

use super::GlweParams;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlweCiphertext<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlweMask<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlweBody<C: Container> {
    pub data: C,
    pub polynomial_size: usize,
}

impl<C: Container> GlweCiphertext<C> {
    pub fn data_len(glwe_params: GlweParams) -> usize {
        glwe_params.polynomial_size * (glwe_params.dimension + 1)
    }

    pub fn new(data: C, glwe_params: GlweParams) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(glwe_params));
        Self { data, glwe_params }
    }

    pub unsafe fn from_raw_parts(data: C::Pointer, glwe_params: GlweParams) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(data, Self::data_len(glwe_params)),
            glwe_params,
        }
    }

    pub fn as_view(&self) -> GlweCiphertext<&[C::Item]> {
        GlweCiphertext {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GlweCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GlweCiphertext {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_mask_and_body(self) -> (GlweMask<C>, GlweBody<C>)
    where
        C: Split,
    {
        let (mask, body) = self
            .data
            .split_at(self.glwe_params.polynomial_size * self.glwe_params.dimension);

        (
            GlweMask {
                data: mask,
                glwe_params: self.glwe_params,
            },
            GlweBody {
                data: body,
                polynomial_size: self.glwe_params.polynomial_size,
            },
        )
    }
    pub fn into_body(self) -> GlweBody<C>
    where
        C: Split,
    {
        self.into_mask_and_body().1
    }
}

impl GlweCiphertext<&mut [u64]> {
    pub fn update_with_wrapping_sub_element_mul(self, other: &[u64], multiplier: u64) {
        for (a, b) in zip_eq(self.data, other) {
            *a = a.wrapping_sub(b.wrapping_mul(multiplier));
        }
    }
}

impl<C: Container> GlweMask<C> {
    pub fn data_len(glwe_params: GlweParams) -> usize {
        glwe_params.polynomial_size * glwe_params.dimension
    }

    pub fn new(data: C, glwe_params: GlweParams) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(glwe_params));
        Self { data, glwe_params }
    }

    pub unsafe fn from_raw_parts(data: C::Pointer, glwe_params: GlweParams) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(data, Self::data_len(glwe_params));
        Self { data, glwe_params }
    }

    pub fn as_view(&self) -> GlweMask<&[C::Item]> {
        GlweMask {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GlweMask<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GlweMask {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn get_polynomial(self, idx: usize) -> C
    where
        C: Split,
    {
        self.data.chunk(
            idx * self.glwe_params.polynomial_size,
            (idx + 1) * self.glwe_params.polynomial_size,
        )
    }
}

impl<C: Container> GlweBody<C> {
    pub fn data_len(polynomial_size: usize) -> usize {
        polynomial_size
    }

    pub fn new(data: C, polynomial_size: usize) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(polynomial_size));
        Self {
            data,
            polynomial_size,
        }
    }

    pub unsafe fn from_raw_parts(data: C::Pointer, polynomial_size: usize) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(data, Self::data_len(polynomial_size)),
            polynomial_size,
        }
    }

    pub fn as_view(&self) -> GlweBody<&[C::Item]> {
        GlweBody {
            data: self.data.as_ref(),
            polynomial_size: self.polynomial_size,
        }
    }

    pub fn as_mut_view(&mut self) -> GlweBody<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GlweBody {
            data: self.data.as_mut(),
            polynomial_size: self.polynomial_size,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}
