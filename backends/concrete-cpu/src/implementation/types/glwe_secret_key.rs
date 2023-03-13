use super::polynomial::Polynomial;
use super::{GlweParams, LweSecretKey};
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct GlweSecretKey<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
}

impl<C: Container> GlweSecretKey<C> {
    pub fn data_len(glwe_params: GlweParams) -> usize {
        glwe_params.lwe_dimension()
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

    pub fn as_view(&self) -> GlweSecretKey<&[C::Item]> {
        GlweSecretKey {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn as_mut_view(&mut self) -> GlweSecretKey<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        GlweSecretKey {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn get_polynomial(self, idx: usize) -> Polynomial<C>
    where
        C: Split,
    {
        Polynomial::new(
            self.data.chunk(
                idx * self.glwe_params.polynomial_size,
                (idx + 1) * self.glwe_params.polynomial_size,
            ),
            self.glwe_params.polynomial_size,
        )
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn as_lwe(self) -> LweSecretKey<C> {
        LweSecretKey::new(self.data, self.glwe_params.lwe_dimension())
    }
}

impl<'a> GlweSecretKey<&'a [u64]> {
    pub fn iter(self) -> impl 'a + DoubleEndedIterator<Item = Polynomial<&'a [u64]>> {
        self.data
            .chunks_exact(self.glwe_params.polynomial_size)
            .map(move |slice| Polynomial::new(slice, self.glwe_params.polynomial_size))
    }
}
