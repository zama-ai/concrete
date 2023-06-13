use dyn_stack::DynStack;

use super::external_product::external_product;
use super::fft::FftView;
use super::types::*;

fn trivial_packing(output: GlweCiphertext<&mut [u64]>, input: LweCiphertext<&[u64]>) {
    debug_assert!(input.lwe_dimension <= output.glwe_params.lwe_dimension());

    let glwe_params = output.glwe_params;
    let polynomial_size = glwe_params.polynomial_size;

    // We retrieve the bodies and masks of the two ciphertexts.
    let (glwe_full_mask, glwe_body) = output.into_data().split_at_mut(glwe_params.lwe_dimension());

    let (glwe_common_mask, additional_mask) = glwe_full_mask.split_at_mut(input.lwe_dimension);

    let (lwe_body, lwe_mask) = input.into_data().split_last().unwrap();

    // We copy the mask (each polynomial is in the wrong order)
    glwe_common_mask.copy_from_slice(lwe_mask);

    additional_mask.fill(0);

    glwe_body.fill(0);

    // We copy the body
    glwe_body[0] = *lwe_body;

    // We compute the number of elements which must be
    // turned into their opposite
    let opposite_count = polynomial_size - 1;

    // We loop through the polynomials (as mut tensors)
    for glwe_mask_poly in glwe_full_mask.chunks_exact_mut(polynomial_size) {
        // We rotate the polynomial properly
        glwe_mask_poly.rotate_right(opposite_count);

        // We compute the opposite of the proper coefficients
        for x in glwe_mask_poly[0..opposite_count].iter_mut() {
            *x = x.wrapping_neg()
        }

        // We reverse the polynomial
        glwe_mask_poly.reverse();
    }
}

/// Performs the external product of `ggsw` and `glwe`, and stores the result in `out`.
pub fn fast_keyswitch(
    out: LweCiphertext<&mut [u64]>,
    ggsw: GgswCiphertext<&[f64]>,
    input: LweCiphertext<&[u64]>,
    fft: FftView<'_>,
    stack: DynStack<'_>,
) -> (Vec<u64>, Vec<u64>) {
    let mut input_glwe_c = vec![0; (ggsw.in_glwe_dimension + 1) * ggsw.polynomial_size];

    let mut input_glwe = GlweCiphertext::new(input_glwe_c.as_mut_slice(), ggsw.in_glwe_params());

    let mut output_glwe_c = vec![0; (ggsw.out_glwe_dimension + 1) * ggsw.polynomial_size];

    let mut output_glwe = GlweCiphertext::new(output_glwe_c.as_mut_slice(), ggsw.out_glwe_params());

    trivial_packing(input_glwe.as_mut_view(), input);

    external_product(
        output_glwe.as_mut_view(),
        ggsw,
        input_glwe.as_view(),
        fft,
        stack,
    );

    output_glwe.as_view().sample_extract(out, 0);

    (input_glwe_c, output_glwe_c)
}

#[cfg(test)]
mod tests {
    use crate::c_api::types::tests::to_generic;
    use crate::implementation::encrypt::encrypt_constant_ggsw;
    use crate::implementation::fast_keyswitch::fast_keyswitch;
    use crate::implementation::fft::Fft;

    use crate::implementation::types::{
        CsprngMut, DecompParams, GgswCiphertext, GlweCiphertext, GlweParams, GlweSecretKey,
        LweCiphertext, LweSecretKey,
    };
    use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
    use concrete_csprng::seeders::Seed;
    use dyn_stack::DynStack;
    use std::mem::MaybeUninit;

    use super::trivial_packing;

    fn init_sk(sk: &mut [u64], csprng: CsprngMut) {
        let len = sk.len();
        LweSecretKey::new(sk, len).fill_with_new_key(csprng)
    }

    #[test]
    fn test_trivial_packing() {
        let in_lwe_dim = 650;

        let out_glwe_params = GlweParams {
            dimension: 2,
            polynomial_size: 512,
        };

        let polynomial_size = out_glwe_params.polynomial_size;

        let mut csprng = SoftwareRandomGenerator::new(Seed(0));
        let mut csprng = to_generic(&mut csprng);

        let mut sk = vec![0; in_lwe_dim];

        init_sk(&mut sk, csprng.as_mut());

        let sk_lwe = LweSecretKey::new(sk.as_slice(), in_lwe_dim);

        let mut sk_glwe = vec![0; out_glwe_params.lwe_dimension()];

        sk_glwe[0..in_lwe_dim].copy_from_slice(sk_lwe.as_view().data);

        let sk_glwe = GlweSecretKey::new(sk_glwe.as_slice(), out_glwe_params);

        let in_ct = 1_u64 << 60;

        let mut in_lwe_ct = vec![0; in_lwe_dim + 1];

        let mut in_lwe_ct = LweCiphertext::new(in_lwe_ct.as_mut_slice(), in_lwe_dim);

        sk_lwe.encrypt_lwe(in_lwe_ct.as_mut_view(), in_ct, 0.0, csprng.as_mut());

        let mut out_glwe_ct = vec![0; (out_glwe_params.dimension + 1) * polynomial_size];

        let mut out_glwe_ct = GlweCiphertext::new(out_glwe_ct.as_mut_slice(), out_glwe_params);

        trivial_packing(out_glwe_ct.as_mut_view(), in_lwe_ct.as_view());

        let decrypted_lwe = sk_lwe.decrypt_lwe(in_lwe_ct.as_view());

        assert!(decrypted_lwe.abs_diff(in_ct) < 1_u64 << 30);

        let decrypted_glwe = sk_glwe.decrypt_glwe(out_glwe_ct.as_view());

        assert_eq!(decrypted_lwe, decrypted_glwe[0]);
    }

    #[test]
    fn fast_ks() {
        let decomp_params = DecompParams {
            level: 3,
            base_log: 10,
        };

        let in_lwe_dim = 1500;

        let out_lwe_dim = 700;

        let polynomial_size = 512;

        let in_glwe_dim = 3;
        let in_glwe_params = GlweParams {
            dimension: in_glwe_dim,
            polynomial_size,
        };

        let out_glwe_dim = 1;
        let out_glwe_params = GlweParams {
            dimension: out_glwe_dim,
            polynomial_size,
        };

        let mut csprng = SoftwareRandomGenerator::new(Seed(0));
        let mut csprng = to_generic(&mut csprng);

        let fft = Fft::new(polynomial_size);

        let mut in_sk = vec![0; in_lwe_dim];

        init_sk(&mut in_sk, csprng.as_mut());

        let in_sk_lwe = LweSecretKey::new(in_sk.as_slice(), in_lwe_dim);

        let mut in_sk_glwe = vec![0; in_glwe_params.lwe_dimension()];

        in_sk_glwe[0..in_lwe_dim].copy_from_slice(in_sk_lwe.as_view().data);

        let in_sk_glwe = GlweSecretKey::new(in_sk_glwe.as_slice(), in_glwe_params);

        let mut out_sk = vec![0; out_lwe_dim];

        init_sk(&mut out_sk, csprng.as_mut());

        let out_sk_lwe = LweSecretKey::new(out_sk.as_slice(), out_lwe_dim);

        let out_sk_glwe =
            GlweSecretKey::new(&out_sk[0..out_glwe_dim * polynomial_size], out_glwe_params);

        let mut ggsw =
            vec![0; polynomial_size * (in_glwe_dim + 1) * (out_glwe_dim + 1) * decomp_params.level];

        let mut ggsw = GgswCiphertext::new(
            ggsw.as_mut_slice(),
            polynomial_size,
            in_glwe_dim,
            out_glwe_dim,
            decomp_params,
        );

        encrypt_constant_ggsw(
            in_sk_glwe,
            out_sk_glwe,
            ggsw.as_mut_view(),
            1,
            0.0,
            csprng.as_mut(),
        );

        let mut ggsw_f =
            vec![
                0.0;
                polynomial_size * (in_glwe_dim + 1) * (out_glwe_dim + 1) * decomp_params.level
            ];

        let mut ggsw_f = GgswCiphertext::new(
            ggsw_f.as_mut_slice(),
            polynomial_size,
            in_glwe_dim,
            out_glwe_dim,
            decomp_params,
        );

        let mut stack = vec![MaybeUninit::new(0_u8); 1000000];
        ggsw_f.as_mut_view().fill_with_forward_fourier(
            ggsw.as_view(),
            fft.as_view(),
            DynStack::new(&mut stack),
        );

        let in_ct = 1_u64 << 62;

        let mut in_lwe_ct = vec![0; in_lwe_dim + 1];

        let mut in_lwe_ct = LweCiphertext::new(in_lwe_ct.as_mut_slice(), in_lwe_dim);

        in_sk_lwe.encrypt_lwe(in_lwe_ct.as_mut_view(), in_ct, 0.0, csprng.as_mut());

        let mut out_lwe_ct = vec![0; out_lwe_dim + 1];

        let mut out_lwe_ct = LweCiphertext::new(out_lwe_ct.as_mut_slice(), out_lwe_dim);

        let mut stack = vec![MaybeUninit::new(0_u8); 1000000];

        let (before, after) = fast_keyswitch(
            out_lwe_ct.as_mut_view(),
            ggsw_f.as_view(),
            in_lwe_ct.as_view(),
            fft.as_view(),
            DynStack::new(&mut stack),
        );

        dbg!(in_sk_glwe.decrypt_glwe(GlweCiphertext::new(before.as_slice(), in_glwe_params))[0]);
        dbg!(out_sk_glwe.decrypt_glwe(GlweCiphertext::new(after.as_slice(), out_glwe_params))[0]);

        let out_ct = out_sk_lwe.decrypt_lwe(out_lwe_ct.as_view());

        assert!(out_ct.abs_diff(in_ct) < (1_u64 << 40));
    }
}
