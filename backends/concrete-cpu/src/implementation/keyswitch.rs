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
mod tests {
    use crate::c_api::types::tests::to_generic;
    use crate::implementation::types::*;
    use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
    use concrete_csprng::seeders::Seed;

    struct KeySet {
        in_dim: usize,
        out_dim: usize,
        in_sk: LweSecretKey<Vec<u64>>,
        out_sk: LweSecretKey<Vec<u64>>,
        ksk: LweKeyswitchKey<Vec<u64>>,
    }

    impl KeySet {
        fn new(
            mut csprng: CsprngMut,
            in_dim: usize,
            out_dim: usize,
            decomp_params: DecompParams,
            key_variance: f64,
        ) -> Self {
            let in_sk = LweSecretKey::new_random(csprng.as_mut(), in_dim);

            let out_sk = LweSecretKey::new_random(csprng.as_mut(), out_dim);

            let ksk_len = (out_dim + 1) * in_dim * decomp_params.level;

            let mut ksk =
                LweKeyswitchKey::new(vec![0_u64; ksk_len], out_dim, in_dim, decomp_params);

            ksk.as_mut_view().fill_with_keyswitch_key(
                in_sk.as_view(),
                out_sk.as_view(),
                key_variance,
                csprng,
            );

            Self {
                in_dim,
                out_dim,
                in_sk,
                out_sk,
                ksk,
            }
        }

        fn keyswitch(&self, csprng: CsprngMut, pt: u64, encryption_variance: f64) -> u64 {
            let mut input = LweCiphertext::zero(self.in_dim);

            let mut output = LweCiphertext::zero(self.out_dim);

            self.in_sk
                .as_view()
                .encrypt_lwe(input.as_mut_view(), pt, encryption_variance, csprng);

            self.ksk
                .as_view()
                .keyswitch_ciphertext(output.as_mut_view(), input.as_view());

            self.out_sk.as_view().decrypt_lwe(output.as_view())
        }
    }

    #[test]
    fn keyswitch_correctness() {
        let mut csprng = SoftwareRandomGenerator::new(Seed(0));

        let keyset = KeySet::new(
            to_generic(&mut csprng),
            1024,
            600,
            DecompParams {
                level: 3,
                base_log: 10,
            },
            0.0000000000000001,
        );

        for _ in 0..100 {
            let input: u64 = u64::from_le_bytes(std::array::from_fn(|_| csprng.next().unwrap()));

            let output = keyset.keyswitch(to_generic(&mut csprng), input, 0.00000000001);

            let diff = output.wrapping_sub(input) as i64;

            assert!((diff as f64).abs() / 2.0_f64.powi(64) < 0.01);
        }
    }
}
