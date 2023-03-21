use crate::implementation::cmux::cmux_scratch;
use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::*;

use super::cmux::cmux;
use super::fft::FftView;
use super::polynomial::{
    update_with_wrapping_monic_monomial_mul, update_with_wrapping_unit_monomial_div,
};
use super::types::*;
use super::zip_eq;

impl<'a> BootstrapKey<&'a [f64]> {
    pub fn blind_rotate_scratch(
        bsk_glwe_params: GlweParams,
        fft: FftView<'_>,
    ) -> Result<StackReq, SizeOverflow> {
        StackReq::try_all_of([
            StackReq::try_new_aligned::<u64>(
                (bsk_glwe_params.dimension + 1) * bsk_glwe_params.polynomial_size,
                CACHELINE_ALIGN,
            )?,
            cmux_scratch(bsk_glwe_params, fft)?,
        ])
    }

    pub fn bootstrap_scratch(
        bsk_glwe_params: GlweParams,
        fft: FftView<'_>,
    ) -> Result<StackReq, SizeOverflow> {
        StackReq::try_all_of([
            StackReq::try_new_aligned::<u64>(
                (bsk_glwe_params.dimension + 1) * bsk_glwe_params.polynomial_size,
                CACHELINE_ALIGN,
            )?,
            Self::blind_rotate_scratch(bsk_glwe_params, fft)?,
        ])
    }

    pub fn blind_rotate(
        self,
        mut lut: GlweCiphertext<&mut [u64]>,
        lwe: LweCiphertext<&[u64]>,
        fft: FftView<'_>,
        mut stack: DynStack<'_>,
    ) {
        let (lwe_body, lwe_mask) = lwe.into_data().split_last().unwrap();

        let lut_poly_size = lut.glwe_params.polynomial_size;
        let modulus_switched_body = pbs_modulus_switch(*lwe_body, lut_poly_size, 0, 0);

        for polynomial in lut.as_mut_view().into_polynomial_list().iter_polynomial() {
            update_with_wrapping_unit_monomial_div(polynomial, modulus_switched_body);
        }

        // We initialize the ct_0 used for the successive cmuxes
        let mut ct0 = lut;

        for (lwe_mask_element, bootstrap_key_ggsw) in zip_eq(lwe_mask.iter(), self.into_ggsw_iter())
        {
            if *lwe_mask_element != 0 {
                let stack = stack.rb_mut();
                // We copy ct_0 to ct_1
                let (mut ct1, stack) = stack
                    .collect_aligned(CACHELINE_ALIGN, ct0.as_view().into_data().iter().copied());
                let mut ct1 = GlweCiphertext::new(&mut *ct1, ct0.glwe_params);

                // We rotate ct_1 by performing ct_1 <- ct_1 * X^{modulus_switched_mask_element}
                let modulus_switched_mask_element =
                    pbs_modulus_switch(*lwe_mask_element, lut_poly_size, 0, 0);
                for polynomial in ct1.as_mut_view().into_polynomial_list().iter_polynomial() {
                    update_with_wrapping_monic_monomial_mul(
                        polynomial,
                        modulus_switched_mask_element,
                    );
                }

                cmux(
                    ct0.as_mut_view(),
                    ct1.as_mut_view(),
                    bootstrap_key_ggsw,
                    fft,
                    stack,
                );
            }
        }
    }

    pub fn bootstrap(
        self,
        lwe_out: LweCiphertext<&mut [u64]>,
        lwe_in: LweCiphertext<&[u64]>,
        accumulator: GlweCiphertext<&[u64]>,
        fft: FftView<'_>,
        stack: DynStack<'_>,
    ) {
        let (mut local_accumulator_data, stack) = stack.collect_aligned(
            CACHELINE_ALIGN,
            accumulator.as_view().into_data().iter().copied(),
        );
        let mut local_accumulator =
            GlweCiphertext::new(&mut *local_accumulator_data, accumulator.glwe_params);
        self.blind_rotate(local_accumulator.as_mut_view(), lwe_in, fft, stack);
        local_accumulator
            .as_view()
            .fill_lwe_with_sample_extraction(lwe_out, 0);
    }
}

/// This function switches modulus for a single coefficient of a ciphertext,
/// only in the context of a PBS
///
/// offset: the number of msb discarded
/// lut_count_log: the right padding
pub fn pbs_modulus_switch(
    input: u64,
    poly_size: usize,
    offset: usize,
    lut_count_log: usize,
) -> usize {
    // First, do the left shift (we discard the offset msb)
    let mut output = input << offset;
    // Start doing the right shift
    output >>= u64::BITS as usize - int_log2(poly_size) - 2 + lut_count_log;
    // Do the rounding
    output += output & 1_u64;
    // Finish the right shift
    output >>= 1;
    // Apply the lsb padding
    output <<= lut_count_log;
    output as usize
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use crate::c_api::types::tests::to_generic;
    use crate::implementation::fft::{Fft, FftView};
    use crate::implementation::types::*;
    use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
    use concrete_csprng::seeders::Seed;
    use dyn_stack::DynStack;

    struct KeySet {
        in_dim: usize,
        glwe_params: GlweParams,
        decomp_params: DecompParams,
        in_sk: LweSecretKey<Vec<u64>>,
        out_sk: LweSecretKey<Vec<u64>>,
        fourier_bsk: BootstrapKey<Vec<f64>>,
        fft: Fft,
        stack: Vec<MaybeUninit<u8>>,
    }

    #[allow(clippy::too_many_arguments)]
    fn new_bsk(
        csprng: CsprngMut,
        in_dim: usize,
        glwe_params: GlweParams,
        decomp_params: DecompParams,
        key_variance: f64,
        in_sk: LweSecretKey<&[u64]>,
        out_sk: GlweSecretKey<&[u64]>,
        fft: FftView,
        stack: DynStack,
    ) -> Vec<f64> {
        let bsk_len = glwe_params.polynomial_size
            * (glwe_params.dimension + 1)
            * (glwe_params.dimension + 1)
            * in_dim
            * decomp_params.level;

        let mut bsk = vec![0_u64; bsk_len];

        BootstrapKey::new(bsk.as_mut_slice(), glwe_params, in_dim, decomp_params)
            .fill_with_new_key_par(in_sk, out_sk, key_variance, csprng);
        let standard = BootstrapKey::new(bsk.as_slice(), glwe_params, in_dim, decomp_params);

        let mut bsk_f = vec![0.; bsk_len];

        let mut fourier =
            BootstrapKey::new(bsk_f.as_mut_slice(), glwe_params, in_dim, decomp_params);

        fourier.fill_with_forward_fourier(standard, fft, stack);

        bsk_f
    }

    impl KeySet {
        fn new(
            mut csprng: CsprngMut,
            in_dim: usize,
            glwe_params: GlweParams,
            decomp_params: DecompParams,
            key_variance: f64,
        ) -> Self {
            let in_sk = LweSecretKey::new_random(csprng.as_mut(), in_dim);

            let out_sk = LweSecretKey::new_random(csprng.as_mut(), glwe_params.lwe_dimension());

            let fft = Fft::new(glwe_params.polynomial_size);

            let mut stack = vec![MaybeUninit::new(0_u8); 100000];

            let bsk_f = new_bsk(
                csprng,
                in_dim,
                glwe_params,
                decomp_params,
                key_variance,
                in_sk.as_view(),
                GlweSecretKey::new(out_sk.data.as_slice(), glwe_params),
                fft.as_view(),
                DynStack::new(&mut stack),
            );

            let fft = Fft::new(glwe_params.polynomial_size);

            let fourier_bsk = BootstrapKey::new(bsk_f, glwe_params, in_dim, decomp_params);

            Self {
                in_dim,
                glwe_params,
                decomp_params,
                in_sk,
                out_sk,
                fourier_bsk,
                fft,
                stack,
            }
        }

        fn bootstrap(
            &mut self,
            csprng: CsprngMut,
            pt: u64,
            encryption_variance: f64,
            lut: GlweCiphertext<&[u64]>,
        ) -> u64 {
            let mut input = LweCiphertext::zero(self.in_dim);

            let mut output = LweCiphertext::zero(self.glwe_params.lwe_dimension());

            self.in_sk
                .as_view()
                .encrypt_lwe(input.as_mut_view(), pt, encryption_variance, csprng);

            assert_eq!(
                lut.data.len(),
                (self.glwe_params.dimension + 1) * self.glwe_params.polynomial_size
            );

            self.fourier_bsk.as_view().bootstrap(
                output.as_mut_view(),
                input.as_view(),
                lut,
                self.fft.as_view(),
                DynStack::new(&mut self.stack),
            );

            self.out_sk.as_view().decrypt_lwe(output.as_view())
        }
    }

    #[test]
    fn bootstrap_correctness() {
        let mut csprng = SoftwareRandomGenerator::new(Seed(0));

        let glwe_dim = 1;

        let log2_poly_size = 10;
        let polynomial_size = 1 << log2_poly_size;

        let glwe_params = GlweParams {
            dimension: glwe_dim,
            polynomial_size,
        };

        let mut keyset = KeySet::new(
            to_generic(&mut csprng),
            600,
            glwe_params,
            DecompParams {
                level: 3,
                base_log: 10,
            },
            0.0000000000000000000001,
        );

        let log2_precision = 4;
        let precision = 1 << log2_precision;

        let lut_case_number: u64 = precision;

        assert_eq!(polynomial_size as u64 % lut_case_number, 0);
        let lut_case_size = polynomial_size as u64 / lut_case_number;

        for _ in 0..100 {
            let lut_index: u64 =
                u64::from_le_bytes(std::array::from_fn(|_| csprng.next().unwrap()))
                    % (2 * precision);

            let lut: Vec<u64> = (0..lut_case_number)
                .map(|_| u64::from_le_bytes(std::array::from_fn(|_| csprng.next().unwrap())))
                .collect();

            let raw_lut: Vec<u64> = (0..glwe_dim)
                .flat_map(|_| (0..polynomial_size).map(|_| 0))
                .chain(
                    lut.iter()
                        .flat_map(|&lut_value| (0..lut_case_size).map(move |_| lut_value)),
                )
                .collect();

            let expected_image = if lut_index < precision {
                lut[lut_index as usize]
            } else {
                lut[(lut_index - precision) as usize].wrapping_neg()
            };

            let pt = (lut_index as f64 + 0.5) / (2. * lut_case_number as f64) * 2.0_f64.powi(64);

            let image = keyset.bootstrap(
                to_generic(&mut csprng),
                pt as u64,
                0.0000000001,
                GlweCiphertext::new(&raw_lut, glwe_params),
            );

            let diff = image.wrapping_sub(expected_image) as i64;

            assert!((diff as f64).abs() / 2.0_f64.powi(64) < 0.01);
        }
    }
}
