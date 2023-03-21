use crate::implementation::{Container, ContainerMut, Split};

use super::LweCiphertext;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct LweCiphertextList<C: Container> {
    pub data: C,
    pub lwe_dimension: usize,
    pub count: usize,
}

impl<C: Container> LweCiphertextList<C> {
    pub fn data_len(lwe_dimension: usize) -> usize {
        lwe_dimension + 1
    }

    pub fn new(data: C, lwe_dimension: usize, count: usize) -> Self {
        debug_assert_eq!(data.len(), (lwe_dimension + 1) * count);
        Self {
            data,
            lwe_dimension,
            count,
        }
    }

    pub unsafe fn from_raw_parts(data: C::Pointer, lwe_dimension: usize, count: usize) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(data, (lwe_dimension + 1) * count),
            lwe_dimension,
            count,
        }
    }

    pub fn as_view(&self) -> LweCiphertextList<&[C::Item]> {
        LweCiphertextList {
            data: self.data.as_ref(),
            lwe_dimension: self.lwe_dimension,
            count: self.count,
        }
    }

    pub fn as_mut_view(&mut self) -> LweCiphertextList<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        LweCiphertextList {
            data: self.data.as_mut(),
            lwe_dimension: self.lwe_dimension,
            count: self.count,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl LweCiphertextList<&mut [u64]> {
    pub fn ciphertext_iter_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = LweCiphertext<&mut [u64]>> {
        self.data
            .chunks_exact_mut(self.lwe_dimension + 1)
            .map(|data| LweCiphertext::new(data, self.lwe_dimension))
    }
}

impl LweCiphertextList<&[u64]> {
    pub fn ciphertext_iter(&self) -> impl DoubleEndedIterator<Item = LweCiphertext<&[u64]>> {
        self.data
            .chunks_exact(self.lwe_dimension + 1)
            .map(|data| LweCiphertext::new(data, self.lwe_dimension))
    }
}
