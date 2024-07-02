use concrete_csprng::generators::SoftwareRandomGenerator;
use tfhe::core_crypto::commons::math::random::{CompressionSeed, Seed};
use tfhe::core_crypto::prelude::*;

use super::csprng::new_dyn_seeder;
use super::types::{EncCsprng, Uint128};
use super::utils::nounwind;
use crate::c_api::types::Parallelism;

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
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let input_key = LweSecretKey::from_container(core::slice::from_raw_parts(
            input_lwe_sk,
            input_lwe_dimension,
        ));
        let output_key = LweSecretKey::from_container(core::slice::from_raw_parts(
            output_lwe_sk,
            output_lwe_dimension,
        ));
        let mut ksk = LweKeyswitchKey::from_container(
            core::slice::from_raw_parts_mut(
                lwe_ksk,
                concrete_cpu_keyswitch_key_size_u64(
                    decomposition_level_count,
                    input_lwe_dimension,
                    output_lwe_dimension,
                ),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_lwe_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        generate_lwe_keyswitch_key(
            &input_key,
            &output_key,
            &mut ksk,
            Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
            &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
        )
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_seeded_lwe_keyswitch_key_u64(
    // keyswitch key
    seeded_lwe_ksk: *mut u64,
    // secret keys
    input_lwe_sk: *const u64,
    output_lwe_sk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    // keyswitch key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    compression_seed: Uint128,
    // noise parameters
    variance: f64,
) {
    nounwind(|| {
        let input_key = LweSecretKey::from_container(core::slice::from_raw_parts(
            input_lwe_sk,
            input_lwe_dimension,
        ));
        let output_key = LweSecretKey::from_container(core::slice::from_raw_parts(
            output_lwe_sk,
            output_lwe_dimension,
        ));

        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let mut seeded_ksk = SeededLweKeyswitchKey::from_container(
            core::slice::from_raw_parts_mut(
                seeded_lwe_ksk,
                concrete_cpu_seeded_keyswitch_key_size_u64(
                    decomposition_level_count,
                    input_lwe_dimension,
                ),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_lwe_dimension).to_lwe_size(),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );

        let mut boxed_seeder = new_dyn_seeder();
        let seeder = boxed_seeder.as_mut();

        generate_seeded_lwe_keyswitch_key(
            &input_key,
            &output_key,
            &mut seeded_ksk,
            Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
            seeder,
        )
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decompress_seeded_lwe_keyswitch_key_u64(
    // keyswitch key
    lwe_ksk: *mut u64,
    // seeded keyswitch key
    seeded_lwe_ksk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_lwe_dimension: usize,
    // keyswitch key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    compression_seed: Uint128,
    // parallelism
    parallelism: Parallelism,
) {
    nounwind(|| {
        let mut output_ksk = LweKeyswitchKey::from_container(
            core::slice::from_raw_parts_mut(
                lwe_ksk,
                concrete_cpu_keyswitch_key_size_u64(
                    decomposition_level_count,
                    input_lwe_dimension,
                    output_lwe_dimension,
                ),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_lwe_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let input_ksk = SeededLweKeyswitchKey::from_container(
            core::slice::from_raw_parts(
                seeded_lwe_ksk,
                concrete_cpu_seeded_keyswitch_key_size_u64(
                    decomposition_level_count,
                    input_lwe_dimension,
                ),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_lwe_dimension).to_lwe_size(),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );
        match parallelism {
            Parallelism::No => {
                decompress_seeded_lwe_keyswitch_key::<_, _, _, SoftwareRandomGenerator>(
                    &mut output_ksk,
                    &input_ksk,
                )
            }
            Parallelism::Rayon => {
                par_decompress_seeded_lwe_keyswitch_key::<_, _, _, SoftwareRandomGenerator>(
                    &mut output_ksk,
                    &input_ksk,
                )
            }
        }
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
        let mut ct_out = LweCiphertext::from_container(
            core::slice::from_raw_parts_mut(ct_out, output_dimension + 1),
            CiphertextModulus::new_native(),
        );
        let ct_in = LweCiphertext::from_container(
            core::slice::from_raw_parts(ct_in, input_dimension + 1),
            CiphertextModulus::new_native(),
        );

        let keyswitch_key = LweKeyswitchKey::from_container(
            core::slice::from_raw_parts(
                keyswitch_key,
                concrete_cpu_keyswitch_key_size_u64(
                    decomposition_level_count,
                    input_dimension,
                    output_dimension,
                ),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );
        keyswitch_lwe_ciphertext(&keyswitch_key, &ct_in, &mut ct_out);
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_keyswitch_key_size_u64(
    decomposition_level_count: usize,
    input_dimension: usize,
    output_dimension: usize,
) -> usize {
    input_dimension
        * lwe_keyswitch_key_input_key_element_encrypted_size(
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(output_dimension).to_lwe_size(),
        )
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_seeded_keyswitch_key_size_u64(
    decomposition_level_count: usize,
    input_dimension: usize,
) -> usize {
    input_dimension
        * seeded_lwe_keyswitch_key_input_key_element_encrypted_size(DecompositionLevelCount(
            decomposition_level_count,
        ))
}
