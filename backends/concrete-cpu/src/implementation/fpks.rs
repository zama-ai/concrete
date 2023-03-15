use super::decomposer::SignedDecomposer;
use super::types::{GlweCiphertext, GlweParams, LweCiphertext, PackingKeyswitchKey};
use super::wop::GlweCiphertextList;
use super::{zip_eq, Container};

impl PackingKeyswitchKey<&[u64]> {
    pub fn private_functional_keyswitch_ciphertext(
        &self,
        mut after: GlweCiphertext<&mut [u64]>,
        before: LweCiphertext<&[u64]>,
    ) {
        debug_assert_eq!(self.glwe_params, after.glwe_params);
        debug_assert_eq!(self.input_dimension, before.lwe_dimension);
        // We reset the output
        after.as_mut_view().into_data().fill_with(|| 0);

        // We instantiate a decomposer
        let decomposer = SignedDecomposer::new(self.decomp_params);

        for (block, input_lwe) in
            zip_eq(self.bit_decomp_iter(), before.as_view().into_data().iter())
        {
            // We decompose
            let rounded = decomposer.closest_representable(*input_lwe);
            let decomp = decomposer.decompose(rounded);

            // Loop over the number of levels:
            // We compute the multiplication of a ciphertext from the private functional
            // keyswitching key with a piece of the decomposition and subtract it to the buffer
            for (level_key_cipher, decomposed) in zip_eq(
                block
                    .data
                    .chunks_exact(
                        (self.glwe_params.dimension + 1) * self.glwe_params.polynomial_size,
                    )
                    .rev(),
                decomp,
            ) {
                after
                    .as_mut_view()
                    .update_with_wrapping_sub_element_mul(level_key_cipher, decomposed.value());
            }
        }
    }
}

pub struct LweKeyBitDecomposition<C: Container> {
    pub data: C,
    pub glwe_params: GlweParams,
    pub count: usize,
}

impl<C: Container> LweKeyBitDecomposition<C> {
    pub fn new(data: C, glwe_params: GlweParams, count: usize) -> Self {
        debug_assert_eq!(
            data.len(),
            (glwe_params.dimension + 1) * glwe_params.polynomial_size * count
        );
        LweKeyBitDecomposition {
            data,
            glwe_params,
            count,
        }
    }

    pub fn into_glwe_list(self) -> GlweCiphertextList<C> {
        GlweCiphertextList {
            data: self.data,
            glwe_params: self.glwe_params,
            count: self.count,
        }
    }
}

impl LweKeyBitDecomposition<&[u64]> {
    pub fn ciphertext_iter(&self) -> impl Iterator<Item = GlweCiphertext<&[u64]>> {
        self.data
            .chunks_exact((self.glwe_params.dimension + 1) * self.glwe_params.polynomial_size)
            .map(move |sub| GlweCiphertext::new(sub, self.glwe_params))
    }
}
impl LweKeyBitDecomposition<&mut [u64]> {
    pub fn ciphertext_iter_mut(&mut self) -> impl Iterator<Item = GlweCiphertext<&mut [u64]>> {
        let glwe_params = self.glwe_params;
        self.data
            .chunks_exact_mut((glwe_params.dimension + 1) * glwe_params.polynomial_size)
            .map(move |sub| GlweCiphertext::new(sub, glwe_params))
    }
}
