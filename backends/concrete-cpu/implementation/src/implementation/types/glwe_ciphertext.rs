use super::polynomial::Polynomial;
use super::polynomial_list::PolynomialList;
use super::GlweParams;
use crate::implementation::{zip_eq, Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlweCiphertext<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
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

    pub fn into_polynomial_list(self) -> PolynomialList<C> {
        PolynomialList {
            data: self.data,
            count: self.glwe_params.dimension + 1,
            polynomial_size: self.glwe_params.polynomial_size,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_mask_and_body(self) -> (PolynomialList<C>, Polynomial<C>)
    where
        C: Split,
    {
        let (mask, body) = self
            .data
            .split_at(self.glwe_params.polynomial_size * self.glwe_params.dimension);

        (
            PolynomialList::new(
                mask,
                self.glwe_params.polynomial_size,
                self.glwe_params.dimension,
            ),
            Polynomial::new(body, self.glwe_params.polynomial_size),
        )
    }
    pub fn into_body(self) -> Polynomial<C>
    where
        C: Split,
    {
        self.into_mask_and_body().1
    }
}

impl GlweCiphertext<&mut [u64]> {
    pub fn update_with_wrapping_sub_element_mul(
        self,
        other: GlweCiphertext<&[u64]>,
        multiplier: u64,
    ) {
        for (a, b) in zip_eq(self.data, other.into_data()) {
            *a = a.wrapping_sub(b.wrapping_mul(multiplier));
        }
    }
}
