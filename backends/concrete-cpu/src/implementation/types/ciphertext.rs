use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct LweCiphertext<C: Container> {
    pub data: C,
    pub lwe_dimension: usize,
}

impl<C: Container> LweCiphertext<C> {
    pub fn data_len(lwe_dimension: usize) -> usize {
        lwe_dimension + 1
    }

    pub fn new(data: C, lwe_dimension: usize) -> Self {
        debug_assert_eq!(data.len(), Self::data_len(lwe_dimension));
        Self {
            data,
            lwe_dimension,
        }
    }

    pub unsafe fn from_raw_parts(data: C::Pointer, lwe_dimension: usize) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(data, Self::data_len(lwe_dimension)),
            lwe_dimension,
        }
    }

    pub fn as_view(&self) -> LweCiphertext<&[C::Item]> {
        LweCiphertext {
            data: self.data.as_ref(),
            lwe_dimension: self.lwe_dimension,
        }
    }

    pub fn as_mut_view(&mut self) -> LweCiphertext<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        LweCiphertext {
            data: self.data.as_mut(),
            lwe_dimension: self.lwe_dimension,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

pub mod test {
    use super::*;

    impl LweCiphertext<Vec<u64>> {
        pub fn zero(dim: usize) -> Self {
            LweCiphertext::new(vec![0; dim + 1], dim)
        }
    }
}
