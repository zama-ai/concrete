use super::*;
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct LevCiphertext<C: Container> {
    pub data: C,
    pub lwe_dimension: usize,
    pub ciphertext_count: usize,
}

impl<C: Container> LevCiphertext<C> {
    pub fn data_len(lwe_dimension: usize, ciphertext_count: usize) -> usize {
        ciphertext_count * (lwe_dimension + 1)
    }

    pub fn new(data: C, lwe_dimension: usize, ciphertext_count: usize) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(lwe_dimension, ciphertext_count));

        Self {
            data,
            lwe_dimension,
            ciphertext_count,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        lwe_dimension: usize,
        ciphertext_count: usize,
    ) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(data, Self::data_len(lwe_dimension, ciphertext_count)),
            lwe_dimension,
            ciphertext_count,
        }
    }

    pub fn as_view(&self) -> LevCiphertext<&[C::Item]> {
        LevCiphertext {
            data: self.data.as_ref(),
            lwe_dimension: self.lwe_dimension,
            ciphertext_count: self.ciphertext_count,
        }
    }

    pub fn as_mut_view(&mut self) -> LevCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        LevCiphertext {
            data: self.data.as_mut(),
            lwe_dimension: self.lwe_dimension,
            ciphertext_count: self.ciphertext_count,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_ciphertext_iter(self) -> impl DoubleEndedIterator<Item = LweCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.ciphertext_count)
            .map(move |slice| LweCiphertext::new(slice, self.lwe_dimension))
    }
}
