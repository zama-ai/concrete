use super::types::{Csprng, CsprngVtable};
use super::utils::nounwind;
use crate::implementation::encrypt::encrypt_constant_ggsw;
#[cfg(target_arch = "x86_64")]
use crate::implementation::types::polynomial::Polynomial;
use crate::implementation::types::{
    CsprngMut, DecompParams, GgswCiphertext, GlweCiphertext, GlweParams, GlweSecretKey,
    LweCiphertext, LweSecretKey,
};
#[cfg(target_arch = "x86_64")]
use concrete_ntt::native_binary64::Plan32;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_secret_key_size_u64(lwe_dimension: usize) -> usize {
    LweSecretKey::<&[u64]>::data_len(lwe_dimension)
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_secret_key_u64(
    sk: *mut u64,
    dimension: usize,
    csprng: *mut Csprng,
    csprng_vtable: *const CsprngVtable,
) {
    nounwind(|| {
        let csprng = CsprngMut::new(csprng, csprng_vtable);

        let sk = LweSecretKey::<&mut [u64]>::from_raw_parts(sk, dimension);

        sk.fill_with_new_key(csprng);
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_encrypt_lwe_ciphertext_u64(
    // secret key
    lwe_sk: *const u64,
    // ciphertext
    lwe_out: *mut u64,
    // plaintext
    input: u64,
    // lwe size
    lwe_dimension: usize,
    // encryption parameters
    variance: f64,
    // csprng
    csprng: *mut Csprng,
    csprng_vtable: *const CsprngVtable,
) {
    nounwind(|| {
        let lwe_sk = LweSecretKey::from_raw_parts(lwe_sk, lwe_dimension);
        let lwe_out = LweCiphertext::from_raw_parts(lwe_out, lwe_dimension);
        lwe_sk.encrypt_lwe(
            lwe_out,
            input,
            variance,
            CsprngMut::new(csprng, csprng_vtable),
        );
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_encrypt_ggsw_ciphertext_u64(
    // secret key
    glwe_sk: *const u64,
    // ciphertext
    ggsw_out: *mut u64,
    // plaintext
    input: u64,
    // glwe size
    glwe_dimension: usize,
    // polynomial_size
    polynomial_size: usize,
    // level
    level: usize,
    // base_log
    base_log: usize,
    // encryption parameters
    variance: f64,
    // csprng
    csprng: *mut Csprng,
    csprng_vtable: *const CsprngVtable,
) {
    nounwind(|| {
        let glwe_params = GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        };
        let glwe_sk = GlweSecretKey::from_raw_parts(glwe_sk, glwe_params);
        let decomp_params = DecompParams { level, base_log };
        let ggsw_out = GgswCiphertext::from_raw_parts(
            ggsw_out,
            polynomial_size,
            glwe_dimension,
            glwe_dimension,
            decomp_params,
        );
        #[cfg(target_arch = "x86_64")]
        let plan = Plan32::try_new(polynomial_size).unwrap();

        #[cfg(target_arch = "x86_64")]
        let mut buffer = vec![0; polynomial_size];
        #[cfg(target_arch = "x86_64")]
        let buffer = Polynomial::new(buffer.as_mut_slice(), polynomial_size);

        encrypt_constant_ggsw(
            glwe_sk,
            glwe_sk,
            ggsw_out,
            input,
            variance,
            CsprngMut::new(csprng, csprng_vtable),
            #[cfg(target_arch = "x86_64")]
            &plan,
            #[cfg(target_arch = "x86_64")]
            buffer,
        );
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decrypt_lwe_ciphertext_u64(
    // secret key
    lwe_sk: *const u64,
    // ciphertext
    lwe_ct_in: *const u64,
    // lwe size
    lwe_dimension: usize,
    // plaintext
    plaintext: *mut u64,
) {
    nounwind(|| {
        let lwe_sk = LweSecretKey::from_raw_parts(lwe_sk, lwe_dimension);
        let lwe_ct_in = LweCiphertext::from_raw_parts(lwe_ct_in, lwe_dimension);
        *plaintext = lwe_sk.decrypt_lwe(lwe_ct_in);
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decrypt_glwe_ciphertext_u64(
    glwe_sk: *const u64,
    polynomial_out: *mut u64,
    glwe_ct_in: *const u64,
    glwe_dimension: usize,
    polynomial_size: usize,
) {
    nounwind(|| {
        let glwe_params = GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        };

        let lwe_sk = GlweSecretKey::from_raw_parts(glwe_sk, glwe_params);
        let polynomial_out = slice::from_raw_parts_mut(polynomial_out, polynomial_size);

        let glwe_ct_in = GlweCiphertext::from_raw_parts(glwe_ct_in, glwe_params);

        lwe_sk.decrypt_glwe_inplace(glwe_ct_in, polynomial_out);
    });
}
