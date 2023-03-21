use super::{DecompParams, GgswCiphertext, GlweParams};
use crate::implementation::fft::FftView;
use crate::implementation::{zip_eq, Container, ContainerMut, Split};
use dyn_stack::{DynStack, ReborrowMut};
#[cfg(feature = "parallel")]
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[readonly::make]
pub struct BootstrapKey<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub input_lwe_dimension: usize,
    pub decomp_params: DecompParams,
}

impl<C: Container> BootstrapKey<C> {
    pub fn data_len(
        glwe_params: GlweParams,
        decomposition_level_count: usize,
        input_lwe_dimension: usize,
    ) -> usize {
        glwe_params.polynomial_size
            * (glwe_params.dimension + 1)
            * (glwe_params.dimension + 1)
            * decomposition_level_count
            * input_lwe_dimension
    }

    pub fn new(
        data: C,
        glwe_params: GlweParams,
        input_lwe_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self {
        debug_assert_eq!(
            data.len(),
            Self::data_len(glwe_params, decomp_params.level, input_lwe_dimension),
        );
        Self {
            data,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
        }
    }

    pub unsafe fn from_raw_parts(
        data: C::Pointer,
        glwe_params: GlweParams,
        input_lwe_dimension: usize,
        decomp_params: DecompParams,
    ) -> Self
    where
        C: Split,
    {
        let data = C::from_raw_parts(
            data,
            Self::data_len(glwe_params, decomp_params.level, input_lwe_dimension),
        );

        Self {
            data,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
        }
    }

    pub fn as_view(&self) -> BootstrapKey<&[C::Item]> {
        BootstrapKey {
            data: self.data.as_ref(),
            glwe_params: self.glwe_params,
            input_lwe_dimension: self.input_lwe_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn as_mut_view(&mut self) -> BootstrapKey<&mut [C::Item]>
    where
        C: ContainerMut,
    {
        BootstrapKey {
            data: self.data.as_mut(),
            glwe_params: self.glwe_params,
            input_lwe_dimension: self.input_lwe_dimension,
            decomp_params: self.decomp_params,
        }
    }

    pub fn into_ggsw_iter(self) -> impl DoubleEndedIterator<Item = GgswCiphertext<C>>
    where
        C: Split,
    {
        self.data
            .split_into(self.input_lwe_dimension)
            .map(move |slice| GgswCiphertext::new(slice, self.glwe_params, self.decomp_params))
    }

    pub fn output_lwe_dimension(&self) -> usize {
        self.glwe_params.lwe_dimension()
    }
}

#[cfg(feature = "parallel")]
impl<'a> BootstrapKey<&'a mut [u64]> {
    pub fn into_ggsw_iter_par(
        self,
    ) -> impl 'a + IndexedParallelIterator<Item = GgswCiphertext<&'a mut [u64]>> {
        debug_assert_eq!(self.data.len() % self.input_lwe_dimension, 0);
        let chunk_size = self.data.len() / self.input_lwe_dimension;

        self.data
            .par_chunks_exact_mut(chunk_size)
            .map(move |slice| GgswCiphertext::new(slice, self.glwe_params, self.decomp_params))
    }
}

impl BootstrapKey<&mut [f64]> {
    pub fn fill_with_forward_fourier(
        &mut self,
        coef_bsk: BootstrapKey<&[u64]>,
        fft: FftView<'_>,
        mut stack: DynStack<'_>,
    ) {
        debug_assert_eq!(self.decomp_params, coef_bsk.decomp_params);
        debug_assert_eq!(self.glwe_params, coef_bsk.glwe_params);
        debug_assert_eq!(self.input_lwe_dimension, coef_bsk.input_lwe_dimension);

        for (a, b) in zip_eq(
            self.as_mut_view().into_ggsw_iter(),
            coef_bsk.into_ggsw_iter(),
        ) {
            a.fill_with_forward_fourier(b, fft, stack.rb_mut());
        }
    }
}
