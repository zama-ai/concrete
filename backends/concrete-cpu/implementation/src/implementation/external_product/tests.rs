use std::mem::MaybeUninit;

use crate::c_api::types::tests::to_generic;
use crate::implementation::encrypt::encrypt_constant_ggsw;
use crate::implementation::fft::Fft;
#[cfg(target_arch = "x86_64")]
use crate::implementation::types::polynomial::Polynomial;
use crate::implementation::types::{
    CsprngMut, DecompParams, GgswCiphertext, GlweCiphertext, GlweParams, GlweSecretKey,
    LweSecretKey,
};
use crate::implementation::zip_eq;
use concrete_csprng::generators::{RandomGenerator, SoftwareRandomGenerator};
use concrete_csprng::seeders::Seed;
#[cfg(target_arch = "x86_64")]
use concrete_ntt::native_binary64::Plan32;
use dyn_stack::DynStack;

use super::external_product;

fn init_sk(sk: &mut [u64], csprng: CsprngMut) {
    let len = sk.len();
    LweSecretKey::new(sk, len).fill_with_new_key(csprng)
}

#[test]
fn test_g_external_product_change_key() {
    let decomp_params = DecompParams {
        level: 3,
        base_log: 10,
    };

    let polynomial_size = 1024;

    let in_glwe_dim = 10;

    let out_glwe_dim = 11;

    let mut csprng = SoftwareRandomGenerator::new(Seed(0));
    let mut csprng = to_generic(&mut csprng);

    let fft = Fft::new(polynomial_size);

    let mut in_sk = vec![0; in_glwe_dim * polynomial_size];

    init_sk(&mut in_sk, csprng.as_mut());

    let in_sk = GlweSecretKey::new(
        in_sk.as_slice(),
        GlweParams {
            dimension: in_glwe_dim,
            polynomial_size,
        },
    );

    let mut out_sk = vec![0; out_glwe_dim * polynomial_size];

    init_sk(&mut out_sk, csprng.as_mut());

    let out_sk = GlweSecretKey::new(
        out_sk.as_slice(),
        GlweParams {
            dimension: out_glwe_dim,
            polynomial_size,
        },
    );

    let mut ggsw =
        vec![0; polynomial_size * (in_glwe_dim + 1) * (out_glwe_dim + 1) * decomp_params.level];

    let mut ggsw = GgswCiphertext::new(
        ggsw.as_mut_slice(),
        polynomial_size,
        in_glwe_dim,
        out_glwe_dim,
        decomp_params,
    );

    #[cfg(target_arch = "x86_64")]
    let plan = Plan32::try_new(polynomial_size).unwrap();

    #[cfg(target_arch = "x86_64")]
    let mut buffer = vec![0; polynomial_size];
    #[cfg(target_arch = "x86_64")]
    let buffer = Polynomial::new(buffer.as_mut_slice(), polynomial_size);

    encrypt_constant_ggsw(
        in_sk,
        out_sk,
        ggsw.as_mut_view(),
        1,
        0.0,
        csprng.as_mut(),
        #[cfg(target_arch = "x86_64")]
        &plan,
        #[cfg(target_arch = "x86_64")]
        buffer,
    );

    let mut ggsw_f =
        vec![0.0; polynomial_size * (in_glwe_dim + 1) * (out_glwe_dim + 1) * decomp_params.level];

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

    let in_ct: Vec<u64> = (0..polynomial_size as u64).collect();

    let in_ct_encoded: Vec<u64> = in_ct
        .iter()
        .map(|a| (*a as f64 * 2.0_f64.powi(64) / polynomial_size as f64) as u64)
        .collect();

    let mut in_glwe_ct = vec![0; (in_glwe_dim + 1) * polynomial_size];

    let mut in_glwe_ct = GlweCiphertext::new(
        in_glwe_ct.as_mut_slice(),
        GlweParams {
            dimension: in_glwe_dim,
            polynomial_size,
        },
    );

    #[cfg(target_arch = "x86_64")]
    let plan = Plan32::try_new(polynomial_size).unwrap();

    #[cfg(target_arch = "x86_64")]
    let mut buffer = vec![0; polynomial_size];
    #[cfg(target_arch = "x86_64")]
    let buffer = Polynomial::new(buffer.as_mut_slice(), polynomial_size);

    in_sk.encrypt_zero_glwe(
        in_glwe_ct.as_mut_view(),
        0.0,
        csprng.as_mut(),
        #[cfg(target_arch = "x86_64")]
        &plan,
        #[cfg(target_arch = "x86_64")]
        buffer,
    );

    for (a, b) in zip_eq(
        in_glwe_ct.as_mut_view().into_body().into_data(),
        in_ct_encoded.iter(),
    ) {
        *a = a.wrapping_add(*b);
    }

    let mut out_glwe_ct = vec![0; (out_glwe_dim + 1) * polynomial_size];

    let mut out_glwe_ct = GlweCiphertext::new(
        out_glwe_ct.as_mut_slice(),
        GlweParams {
            dimension: out_glwe_dim,
            polynomial_size,
        },
    );

    let mut stack = vec![MaybeUninit::new(0_u8); 1000000];

    external_product(
        out_glwe_ct.as_mut_view(),
        ggsw_f.as_view(),
        in_glwe_ct.as_view(),
        fft.as_view(),
        DynStack::new(&mut stack),
    );

    let mut out_ct = out_sk.decrypt_glwe(out_glwe_ct.as_view());

    for i in out_ct.iter_mut() {
        *i = (((*i as f64 * polynomial_size as f64) / 2.0_f64.powi(64) + 0.5) as u64)
            % polynomial_size as u64;
    }

    assert_eq!(out_ct, in_ct);
}
