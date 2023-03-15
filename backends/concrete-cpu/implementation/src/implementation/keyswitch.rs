use super::decomposer::SignedDecomposer;
use super::types::*;
use super::zip_eq;

impl LweKeyswitchKey<&[u64]> {
    pub fn keyswitch_ciphertext(
        self,
        after: LweCiphertext<&mut [u64]>,
        before: LweCiphertext<&[u64]>,
    ) {
        let after = after.into_data();
        let before = before.into_data();

        // We reset the output
        after.fill(0);

        // We copy the body
        *after.last_mut().unwrap() = *before.last().unwrap();

        // We instantiate a decomposer
        let decomposer = SignedDecomposer::new(self.decomp_params);

        let mask_len = before.len() - 1;

        for (block, before_mask) in zip_eq(self.into_lev_ciphertexts(), &before[..mask_len]) {
            let mask_rounded = decomposer.closest_representable(*before_mask);
            let decomp = decomposer.decompose(mask_rounded);

            // loop over the number of levels
            for (level_key_cipher, decomposed) in zip_eq(
                block.into_data().chunks(self.output_dimension + 1).rev(),
                decomp,
            ) {
                let val = decomposed.value();
                for (a, &b) in zip_eq(after.iter_mut(), level_key_cipher) {
                    *a = a.wrapping_sub(b.wrapping_mul(val))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
