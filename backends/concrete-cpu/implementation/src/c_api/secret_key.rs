use concrete_csprng::generators::SoftwareRandomGenerator;
use tfhe::core_crypto::commons::math::random::{CompressionSeed, Seed};
use tfhe::core_crypto::prelude::*;

use super::csprng::new_dyn_seeder;
use super::types::{EncCsprng, SecCsprng, Uint128};
use super::utils::nounwind;
use core::slice;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_secret_key_u64(
    sk: *mut u64,
    dimension: usize,
    csprng: *mut SecCsprng,
) {
    nounwind(|| {
        let mut sk = LweSecretKey::from_container(slice::from_raw_parts_mut(
            sk,
            concrete_cpu_lwe_secret_key_size_u64(dimension),
        ));
        tfhe::core_crypto::algorithms::generate_binary_lwe_secret_key(
            &mut sk,
            &mut *(csprng as *mut SecretRandomGenerator<SoftwareRandomGenerator>),
        );
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
    // lwe dimension
    lwe_dimension: usize,
    // encryption parameters
    variance: f64,
    // csprng
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let lwe_sk = LweSecretKey::from_container(slice::from_raw_parts(
            lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(lwe_dimension),
        ));
        let mut lwe_out = LweCiphertext::from_container(
            slice::from_raw_parts_mut(lwe_out, concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension)),
            CiphertextModulus::new_native(),
        );
        encrypt_lwe_ciphertext(
            &lwe_sk,
            &mut lwe_out,
            Plaintext(input),
            Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
            &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
        );
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_encrypt_seeded_lwe_ciphertext_u64(
    // secret key
    lwe_sk: *const u64,
    // seeded ciphertext
    seeded_lwe_out: *mut u64,
    // plaintext
    input: u64,
    // lwe dimension
    lwe_dimension: usize,
    // compression seed
    compression_seed: Uint128,
    // encryption parameters
    variance: f64,
) {
    nounwind(|| {
        let lwe_sk = LweSecretKey::from_container(slice::from_raw_parts(
            lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(lwe_dimension),
        ));

        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let mut seeded_lwe_ciphertext = SeededLweCiphertext::from_scalar(
            *seeded_lwe_out,
            LweDimension(lwe_dimension).to_lwe_size(),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );

        let mut boxed_seeder = new_dyn_seeder();
        let seeder = boxed_seeder.as_mut();
        encrypt_seeded_lwe_ciphertext(
            &lwe_sk,
            &mut seeded_lwe_ciphertext,
            Plaintext(input),
            Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
            seeder,
        );
        *seeded_lwe_out = seeded_lwe_ciphertext.into_scalar();
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
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let glwe_sk = GlweSecretKey::from_container(
            slice::from_raw_parts(
                glwe_sk,
                concrete_cpu_glwe_secret_key_size_u64(glwe_dimension, polynomial_size),
            ),
            PolynomialSize(polynomial_size),
        );
        let mut ggsw_out = GgswCiphertext::from_container(
            slice::from_raw_parts_mut(
                ggsw_out,
                concrete_cpu_ggsw_ciphertext_size_u64(glwe_dimension, polynomial_size, level),
            ),
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionBaseLog(base_log),
            CiphertextModulus::new_native(),
        );
        encrypt_constant_ggsw_ciphertext(
            &glwe_sk,
            &mut ggsw_out,
            Cleartext(input),
            Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
            &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
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
        let lwe_sk = LweSecretKey::from_container(slice::from_raw_parts(
            lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(lwe_dimension),
        ));
        let lwe_ct_in = LweCiphertext::from_container(
            slice::from_raw_parts(
                lwe_ct_in,
                concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension),
            ),
            CiphertextModulus::new_native(),
        );
        *plaintext = decrypt_lwe_ciphertext(&lwe_sk, &lwe_ct_in).0;
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decompress_seeded_lwe_ciphertext_u64(
    // ciphertext
    lwe_out: *mut u64,
    // seeded ciphertext
    seeded_lwe_in: *const u64,
    // lwe dimension
    lwe_dimension: usize,
    // compression seed
    compression_seed: Uint128,
) {
    nounwind(|| {
        let mut lwe_out = LweCiphertext::from_container(
            slice::from_raw_parts_mut(lwe_out, concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension)),
            CiphertextModulus::new_native(),
        );

        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let seeded_lwe_in = SeededLweCiphertext::from_scalar(
            *seeded_lwe_in,
            LweDimension(lwe_dimension).to_lwe_size(),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );

        decompress_seeded_lwe_ciphertext::<_, _, SoftwareRandomGenerator>(
            &mut lwe_out,
            &seeded_lwe_in,
        )
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_serialize_lwe_secret_key_u64(
    lwe_sk: *const u64,
    lwe_dimension: usize,
    out_buffer: *mut u8,
    out_buffer_len: usize,
) -> usize {
    let lwe_sk: LweSecretKey<Vec<u64>> = LweSecretKey::from_container(
        slice::from_raw_parts(lwe_sk, concrete_cpu_lwe_secret_key_size_u64(lwe_dimension)).to_vec(),
    );

    super::utils::safe_serialize(&lwe_sk, out_buffer, out_buffer_len)
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_unserialize_lwe_secret_key_u64(
    buffer: *const u8,
    buffer_len: usize,
    lwe_sk: *mut u64,
    lwe_sk_size: usize,
) -> usize {
    let sk: LweSecretKey<Vec<u64>> = super::utils::safe_deserialize(buffer, buffer_len);
    let container = sk.into_container();
    assert!(container.len() <= lwe_sk_size);
    let lwe_sk_slice = slice::from_raw_parts_mut(lwe_sk, lwe_sk_size);
    lwe_sk_slice.copy_from_slice(container.as_slice());
    container.len()
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_serialize_glwe_secret_key_u64(
    glwe_sk: *const u64,
    glwe_dimension: usize,
    polynomial_size: usize,
    out_buffer: *mut u8,
    out_buffer_len: usize,
) -> usize {
    let glwe_sk: GlweSecretKey<Vec<u64>> = GlweSecretKey::from_container(
        slice::from_raw_parts(
            glwe_sk,
            concrete_cpu_glwe_secret_key_size_u64(glwe_dimension, polynomial_size),
        )
        .to_vec(),
        PolynomialSize(polynomial_size),
    );

    super::utils::safe_serialize(&glwe_sk, out_buffer, out_buffer_len)
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_unserialize_glwe_secret_key_u64(
    buffer: *const u8,
    buffer_len: usize,
    glwe_sk: *mut u64,
    glwe_sk_size: usize,
) -> usize {
    let sk: GlweSecretKey<Vec<u64>> = super::utils::safe_deserialize(buffer, buffer_len);
    assert!(sk.glwe_dimension().0 == 1);
    let container = sk.into_container();
    assert!(container.len() <= glwe_sk_size);
    let glwe_sk_slice = slice::from_raw_parts_mut(glwe_sk, glwe_sk_size);
    glwe_sk_slice.copy_from_slice(container.as_slice());
    container.len()
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decrypt_glwe_ciphertext_u64(
    glwe_sk: *const u64,
    output: *mut u64,
    glwe_ct_in: *const u64,
    glwe_dimension: usize,
    polynomial_size: usize,
) {
    nounwind(|| {
        let glwe_sk = GlweSecretKey::from_container(
            slice::from_raw_parts(
                glwe_sk,
                concrete_cpu_glwe_secret_key_size_u64(glwe_dimension, polynomial_size),
            ),
            PolynomialSize(polynomial_size),
        );

        let glwe_ct_in = GlweCiphertext::from_container(
            slice::from_raw_parts(
                glwe_ct_in,
                concrete_cpu_glwe_ciphertext_size_u64(glwe_dimension, polynomial_size),
            ),
            PolynomialSize(polynomial_size),
            CiphertextModulus::new_native(),
        );

        let mut output =
            PlaintextList::from_container(slice::from_raw_parts_mut(output, polynomial_size));

        decrypt_glwe_ciphertext(&glwe_sk, &glwe_ct_in, &mut output);
    });
}

#[no_mangle]
pub extern "C" fn concrete_cpu_lwe_secret_key_size_u64(lwe_dimension: usize) -> usize {
    lwe_dimension
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_glwe_secret_key_size_u64(
    lwe_dimension: usize,
    polynomial_size: usize,
) -> usize {
    lwe_dimension * polynomial_size
}

#[no_mangle]
pub extern "C" fn concrete_cpu_lwe_secret_key_buffer_size_u64(lwe_dimension: usize) -> usize {
    let metadata = core::mem::size_of::<LweSecretKey<&[u64]>>();
    metadata + concrete_cpu_lwe_secret_key_size_u64(lwe_dimension) * 8 /*u64*/
    + 100 /*serialization headers (fragile)*/
}

#[no_mangle]
pub extern "C" fn concrete_cpu_glwe_secret_key_buffer_size_u64(
    glwe_dimension: usize,
    polynomial_size: usize,
) -> usize {
    let metadata = core::mem::size_of::<GlweSecretKey<&[u64]>>();
    metadata + glwe_dimension * polynomial_size * 8 /*u64*/
    + 100 /*serialization headers (fragile)*/
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension: usize) -> usize {
    lwe_dimension + 1
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_glwe_ciphertext_size_u64(
    glwe_dimension: usize,
    polynomial_size: usize,
) -> usize {
    glwe_ciphertext_size(
        GlweDimension(glwe_dimension).to_glwe_size(),
        PolynomialSize(polynomial_size),
    )
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_ggsw_ciphertext_size_u64(
    glwe_dimension: usize,
    polynomial_size: usize,
    decomposition_level_count: usize,
) -> usize {
    ggsw_ciphertext_size(
        GlweDimension(glwe_dimension).to_glwe_size(),
        PolynomialSize(polynomial_size),
        DecompositionLevelCount(decomposition_level_count),
    )
}
