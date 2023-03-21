use super::CsprngMut;
use crate::implementation::{Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct LweSecretKey<C: Container> {
    pub data: C,
    pub lwe_dimension: usize,
}

impl<C: Container> LweSecretKey<C> {
    pub fn data_len(lwe_dimension: usize) -> usize {
        lwe_dimension
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

    pub fn as_view(&self) -> LweSecretKey<&[C::Item]> {
        LweSecretKey {
            data: self.data.as_ref(),
            lwe_dimension: self.lwe_dimension,
        }
    }

    pub fn as_mut_view(&mut self) -> LweSecretKey<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        LweSecretKey {
            data: self.data.as_mut(),
            lwe_dimension: self.lwe_dimension,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }
}

impl LweSecretKey<&mut [u64]> {
    pub fn fill_with_new_key(self, mut csprng: CsprngMut<'_, '_>) {
        for sk_bit in self.data {
            let mut bytes = [0_u8; 1];

            let success_count = csprng.as_mut().next_bytes(&mut bytes);
            if success_count == 0 {
                panic!("Csprng failed to generate random bytes");
            }

            *sk_bit = (bytes[0] & 1) as u64;
        }
    }
}

pub mod test {
    use super::*;
    use crate::implementation::types::CsprngMut;

    impl LweSecretKey<Vec<u64>> {
        pub fn new_random(csprng: CsprngMut, dim: usize) -> Self {
            let mut sk = LweSecretKey::new(vec![0; dim], dim);

            sk.as_mut_view().fill_with_new_key(csprng);

            sk
        }
    }
}
