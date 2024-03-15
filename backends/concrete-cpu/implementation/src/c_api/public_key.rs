use concrete_csprng::generators::SoftwareRandomGenerator;
use tfhe::core_crypto::prelude::*;

use super::secret_key::{
    concrete_cpu_lwe_ciphertext_size_u64, concrete_cpu_lwe_secret_key_size_u64,
};
use super::types::{EncCsprng, SecCsprng};
use super::utils::nounwind;
use core::slice;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_public_key_size_u64(
    lwe_dimension: usize,
    zero_encryption_count: usize,
) -> usize {
    LweDimension(lwe_dimension).to_lwe_size().0 * zero_encryption_count
}

// https://github.com/zama-ai/tfhe-rs/blob/bcbab1195057b986e1f6d0f2b630169703e95609/tfhe/src/shortint/engine/public_side.rs#L10
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_public_key_zero_encryption_count(
    lwe_dimension: usize,
) -> usize {
    const LOG2_Q_64: usize = 64;
    LwePublicKeyZeroEncryptionCount(LweDimension(lwe_dimension).to_lwe_size().0 * LOG2_Q_64 + 128).0
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_lwe_public_key_u64(
    lwe_secret_key_buffer: *const u64,
    lwe_public_key_buffer: *mut u64,
    lwe_dimension: usize,
    zero_encryption_count: usize,
    variance: f64,
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let mut lwe_public_key = LwePublicKey::from_container(
            slice::from_raw_parts_mut(
                lwe_public_key_buffer,
                concrete_cpu_lwe_public_key_size_u64(lwe_dimension, zero_encryption_count),
            ),
            LweDimension(lwe_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );
        let lwe_secret_key = LweSecretKey::from_container(slice::from_raw_parts(
            lwe_secret_key_buffer,
            concrete_cpu_lwe_secret_key_size_u64(lwe_dimension),
        ));
        tfhe::core_crypto::algorithms::generate_lwe_public_key(
            &lwe_secret_key,
            &mut lwe_public_key,
            Variance::from_variance(variance),
            &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_encrypt_lwe_ciphertext_with_lwe_public_key_u64(
    lwe_public_key_buffer: *const u64,
    lwe_ciphertext_buffer: *mut u64,
    input: u64,
    lwe_dimension: usize,
    zero_encryption_count: usize,
    csprng: *mut SecCsprng,
) {
    nounwind(|| {
        let lwe_public_key = LwePublicKey::from_container(
            slice::from_raw_parts(
                lwe_public_key_buffer,
                concrete_cpu_lwe_public_key_size_u64(lwe_dimension, zero_encryption_count),
            ),
            LweDimension(lwe_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );
        let mut lwe_ciphertext = LweCiphertext::from_container(
            slice::from_raw_parts_mut(
                lwe_ciphertext_buffer,
                concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension),
            ),
            CiphertextModulus::new_native(),
        );
        encrypt_lwe_ciphertext_with_public_key(
            &lwe_public_key,
            &mut lwe_ciphertext,
            Plaintext(input),
            &mut *(csprng as *mut SecretRandomGenerator<SoftwareRandomGenerator>),
        )
    });
}
