use super::*;
use crate::implementation::{zip_eq, Container, ContainerMut, Split};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct PackingKeyswitchKeyList<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub input_dimension: usize,
    pub decomp_params: DecompParams,
    pub count: usize,
}

impl<C: Container> PackingKeyswitchKeyList<C> {
    pub fn data_len(
        glwe_params: GlweParams,
        decomposition_level_count: usize,
        input_dimension: usize,
        count: usize,
    ) -> usize {
        (input_dimension + 1)
            * decomposition_level_count
            * (glwe_params.dimension + 1)
            * glwe_params.polynomial_size
            * count
    }

    pub fn new(
        data: C,
        glwe_params: GlweParams,
        input_dimension: usize,
        decomp_params: DecompParams,
        count: usize,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(glwe_params, decomp_params.level, input_dimension, count),
        );

        Self {
            data,
            glwe_params,
            input_dimension,
            decomp_params,
            count,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        input_dimension: usize,
        decomp_params: DecompParams,
        count: usize,
    ) -> Self
    where
        C: Split,
    {
        Self {
            data: C::from_raw_parts(
                data,
                Self::data_len(glwe_params, decomp_params.level, input_dimension, count),
            ),
            glwe_params,
            input_dimension,
            decomp_params,
            count,
        }
    }

    pub fn into_data(self) -> C {
        self.data
    }

    pub fn into_ppksk_key(self) -> impl DoubleEndedIterator<Item = PackingKeyswitchKey<C>>
    where
        C: Split,
    {
        let glwe_params = self.glwe_params;
        let input_dimension = self.input_dimension;
        let decomp_params = self.decomp_params;
        let count = self.count;
        self.into_data().split_into(count).map(move |slice| {
            PackingKeyswitchKey::new(slice, glwe_params, input_dimension, decomp_params)
        })
    }

    pub fn as_mut_view(&mut self) -> PackingKeyswitchKeyList<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        PackingKeyswitchKeyList {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            input_dimension: self.input_dimension,
            decomp_params: self.decomp_params,
            count: self.count,
        }
    }
}

impl PackingKeyswitchKeyList<&mut [u64]> {
    pub fn fill_with_fpksk_for_circuit_bootstrap(
        &mut self,
        input_lwe_key: &LweSecretKey<&[u64]>,
        output_glwe_key: &GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut,
    ) {
        let glwe_params = output_glwe_key.glwe_params;
        let polynomial_size = glwe_params.polynomial_size;
        debug_assert_eq!(self.count, output_glwe_key.glwe_params.dimension + 1);

        let mut last_polynomial = vec![0; polynomial_size];
        // We apply the x -> -x function so instead of putting one in the first coeff of the
        // polynomial, we put Scalar::MAX == - Sclar::One so that we can use a single function in
        // the loop avoiding branching
        last_polynomial[0] = u64::MAX;

        for (mut fpksk, polynomial_to_encrypt) in zip_eq(
            self.as_mut_view().into_ppksk_key(),
            output_glwe_key
                .data
                .chunks_exact(polynomial_size)
                .chain(std::iter::once(last_polynomial.as_slice())),
        ) {
            fpksk.fill_with_private_functional_packing_keyswitch_key(
                input_lwe_key,
                output_glwe_key,
                variance,
                csprng.as_mut(),
                |x: u64| x.wrapping_neg(),
                polynomial_to_encrypt,
            );
        }
    }
    pub fn fill_with_fpksk_for_circuit_bootstrap_par(
        &mut self,
        input_lwe_key: &LweSecretKey<&[u64]>,
        output_glwe_key: &GlweSecretKey<&[u64]>,
        variance: f64,
        mut csprng: CsprngMut,
    ) {
        let glwe_params = output_glwe_key.glwe_params;
        let polynomial_size = glwe_params.polynomial_size;
        debug_assert_eq!(self.count, output_glwe_key.glwe_params.dimension + 1);

        let mut last_polynomial = vec![0; polynomial_size];
        // We apply the x -> -x function so instead of putting one in the first coeff of the
        // polynomial, we put Scalar::MAX == - Sclar::One so that we can use a single function in
        // the loop avoiding branching
        last_polynomial[0] = u64::MAX;

        for (mut fpksk, polynomial_to_encrypt) in zip_eq(
            self.as_mut_view().into_ppksk_key(),
            output_glwe_key
                .data
                .chunks_exact(polynomial_size)
                .chain(std::iter::once(last_polynomial.as_slice())),
        ) {
            fpksk.fill_with_private_functional_packing_keyswitch_key_par(
                input_lwe_key,
                output_glwe_key,
                variance,
                csprng.as_mut(),
                |x: u64| x.wrapping_neg(),
                polynomial_to_encrypt,
            );
        }
    }
}
