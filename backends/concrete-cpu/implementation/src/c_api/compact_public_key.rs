use concrete_csprng::generators::SoftwareRandomGenerator;
use tfhe::core_crypto::prelude::*;

use super::secret_key::{
    concrete_cpu_lwe_ciphertext_size_u64, concrete_cpu_lwe_secret_key_size_u64,
};
use super::types::{EncCsprng, SecCsprng};
use super::utils::nounwind;
use core::slice;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_compact_public_key_size_u64(
    lwe_dimension: usize,
) -> usize {
    lwe_dimension * 2
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_lwe_compact_public_key_u64(
    lwe_secret_key_buffer: *const u64,
    lwe_public_key_buffer: *mut u64,
    lwe_dimension: usize,
    variance: f64,
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let mut lwe_compact_public_key = LweCompactPublicKey::from_container(
            slice::from_raw_parts_mut(
                lwe_public_key_buffer,
                concrete_cpu_lwe_compact_public_key_size_u64(lwe_dimension),
            ),
            CiphertextModulus::new_native(),
        );
        let lwe_secret_key = LweSecretKey::from_container(slice::from_raw_parts(
            lwe_secret_key_buffer,
            concrete_cpu_lwe_secret_key_size_u64(lwe_dimension),
        ));
        tfhe::core_crypto::algorithms::generate_lwe_compact_public_key(
            &lwe_secret_key,
            &mut lwe_compact_public_key,
            Variance::from_variance(variance),
            &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_encrypt_lwe_ciphertext_with_compact_lwe_public_key_u64(
    lwe_compact_public_key_buffer: *const u64,
    lwe_ciphertext_buffer: *mut u64,
    input: u64,
    lwe_dimension: usize,
    variance: f64,
    secret_csprng: *mut SecCsprng,
    encryption_csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let lwe_compact_public_key = LweCompactPublicKey::from_container(
            slice::from_raw_parts(
                lwe_compact_public_key_buffer,
                concrete_cpu_lwe_compact_public_key_size_u64(lwe_dimension),
            ),
            CiphertextModulus::new_native(),
        );
        let mut lwe_ciphertext = LweCiphertext::from_container(
            slice::from_raw_parts_mut(
                lwe_ciphertext_buffer,
                concrete_cpu_lwe_ciphertext_size_u64(lwe_dimension),
            ),
            CiphertextModulus::new_native(),
        );
        //
        encrypt_lwe_ciphertext_with_compact_public_key(
            &lwe_compact_public_key,
            &mut lwe_ciphertext,
            Plaintext(input),
            Variance::from_variance(variance),
            Variance::from_variance(variance),
            &mut *(secret_csprng as *mut SecretRandomGenerator<SoftwareRandomGenerator>),
            &mut *(encryption_csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
        )
    });
}
