use super::{
    types::{Csprng, CsprngVtable},
    utils::nounwind,
};
use crate::implementation::types::*;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_lwe_keyswitch_key_u64(
    // keyswitch key
    lwe_ksk: *mut u64,
    // secret keys
    input_lwe_sk: *const u64,
    output_lwe_sk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    // keyswitch key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    // noise parameters
    variance: f64,
    // csprng
    csprng: *mut Csprng,
    csprng_vtable: *const CsprngVtable,
) {
    nounwind(|| {
        let input_key = LweSecretKey::from_raw_parts(input_lwe_sk, input_lwe_dimension);
        let output_key = LweSecretKey::from_raw_parts(output_lwe_sk, output_lwe_dimension);
        let ksk = LweKeyswitchKey::from_raw_parts(
            lwe_ksk,
            output_lwe_dimension,
            input_lwe_dimension,
            DecompParams {
                level: decomposition_level_count,
                base_log: decomposition_base_log,
            },
        );

        ksk.fill_with_keyswitch_key(
            input_key,
            output_key,
            variance,
            CsprngMut::new(csprng, csprng_vtable),
        );
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_keyswitch_lwe_ciphertext_u64(
    // ciphertexts
    ct_out: *mut u64,
    ct_in: *const u64,
    // keyswitch key
    keyswitch_key: *const u64,
    // keyswitch parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    input_dimension: usize,
    output_dimension: usize,
) {
    nounwind(|| {
        let ct_out = LweCiphertext::from_raw_parts(ct_out, output_dimension);
        let ct_in = LweCiphertext::from_raw_parts(ct_in, input_dimension);

        let keyswitch_key = LweKeyswitchKey::from_raw_parts(
            keyswitch_key,
            output_dimension,
            input_dimension,
            DecompParams {
                level: decomposition_level_count,
                base_log: decomposition_base_log,
            },
        );

        keyswitch_key.keyswitch_ciphertext(ct_out, ct_in);
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_keyswitch_key_size_u64(
    decomposition_level_count: usize,
    _decomposition_base_log: usize,
    input_dimension: usize,
    output_dimension: usize,
) -> usize {
    LweKeyswitchKey::<&[u64]>::data_len(
        output_dimension,
        decomposition_level_count,
        input_dimension,
    )
}
