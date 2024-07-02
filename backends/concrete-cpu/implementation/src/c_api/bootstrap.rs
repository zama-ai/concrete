use concrete_csprng::generators::SoftwareRandomGenerator;
use concrete_fft::c64;
use tfhe::core_crypto::commons::math::random::{CompressionSeed, Seed};
use tfhe::core_crypto::prelude::*;

use crate::c_api::types::{EncCsprng, Parallelism, ScratchStatus, Uint128};
use core::slice;
use dyn_stack::PodStack;

use super::csprng::new_dyn_seeder;
use super::secret_key::{
    concrete_cpu_glwe_ciphertext_size_u64, concrete_cpu_glwe_secret_key_size_u64,
    concrete_cpu_lwe_secret_key_size_u64,
};
use super::utils::nounwind;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_lwe_bootstrap_key_u64(
    // bootstrap key
    lwe_bsk: *mut u64,
    // secret keys
    input_lwe_sk: *const u64,
    output_glwe_sk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    // bootstrap key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    // noise parameters
    variance: f64,
    // parallelism
    parallelism: Parallelism,
    // csprng
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let mut bsk = LweBootstrapKey::from_container(
            slice::from_raw_parts_mut(
                lwe_bsk,
                concrete_cpu_bootstrap_key_size_u64(
                    decomposition_level_count,
                    output_glwe_dimension,
                    output_polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            GlweDimension(output_glwe_dimension).to_glwe_size(),
            PolynomialSize(output_polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            CiphertextModulus::new_native(),
        );

        let lwe_sk = LweSecretKey::from_container(slice::from_raw_parts(
            input_lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(input_lwe_dimension),
        ));
        let glwe_sk = GlweSecretKey::from_container(
            slice::from_raw_parts(
                output_glwe_sk,
                concrete_cpu_glwe_secret_key_size_u64(
                    output_glwe_dimension,
                    output_polynomial_size,
                ),
            ),
            PolynomialSize(output_polynomial_size),
        );

        match parallelism {
            Parallelism::No => generate_lwe_bootstrap_key(
                &lwe_sk,
                &glwe_sk,
                &mut bsk,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
            ),
            Parallelism::Rayon => par_generate_lwe_bootstrap_key(
                &lwe_sk,
                &glwe_sk,
                &mut bsk,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
            ),
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_seeded_lwe_bootstrap_key_u64(
    // seeded bootstrap key
    seeded_lwe_bsk: *mut u64,
    // secret keys
    input_lwe_sk: *const u64,
    output_glwe_sk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    // seeded bootstrap key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    compression_seed: Uint128,
    // noise parameters
    variance: f64,
    // parallelism
    parallelism: Parallelism,
) {
    nounwind(|| {
        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let mut bsk = SeededLweBootstrapKey::from_container(
            slice::from_raw_parts_mut(
                seeded_lwe_bsk,
                concrete_cpu_seeded_bootstrap_key_size_u64(
                    decomposition_level_count,
                    output_glwe_dimension,
                    output_polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            GlweDimension(output_glwe_dimension).to_glwe_size(),
            PolynomialSize(output_polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );

        let lwe_sk = LweSecretKey::from_container(slice::from_raw_parts(
            input_lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(input_lwe_dimension),
        ));
        let glwe_sk = GlweSecretKey::from_container(
            slice::from_raw_parts(
                output_glwe_sk,
                concrete_cpu_glwe_secret_key_size_u64(
                    output_glwe_dimension,
                    output_polynomial_size,
                ),
            ),
            PolynomialSize(output_polynomial_size),
        );

        let mut boxed_seeder = new_dyn_seeder();
        let seeder = boxed_seeder.as_mut();

        match parallelism {
            Parallelism::No => generate_seeded_lwe_bootstrap_key(
                &lwe_sk,
                &glwe_sk,
                &mut bsk,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                seeder,
            ),
            Parallelism::Rayon => par_generate_seeded_lwe_bootstrap_key(
                &lwe_sk,
                &glwe_sk,
                &mut bsk,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                seeder,
            ),
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_decompress_seeded_lwe_bootstrap_key_u64(
    // bootstrap key
    lwe_bsk: *mut u64,
    // seeded bootstrap key
    seeded_lwe_bsk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    // bootstrap key parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    compression_seed: Uint128,
    // parallelism
    parallelism: Parallelism,
) {
    nounwind(|| {
        let mut output_bsk = LweBootstrapKey::from_container(
            slice::from_raw_parts_mut(
                lwe_bsk,
                concrete_cpu_bootstrap_key_size_u64(
                    decomposition_level_count,
                    output_glwe_dimension,
                    output_polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            GlweDimension(output_glwe_dimension).to_glwe_size(),
            PolynomialSize(output_polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            CiphertextModulus::new_native(),
        );

        let seed = Seed(u128::from_le_bytes(compression_seed.little_endian_bytes));

        let input_bsk = SeededLweBootstrapKey::from_container(
            slice::from_raw_parts(
                seeded_lwe_bsk,
                concrete_cpu_seeded_bootstrap_key_size_u64(
                    decomposition_level_count,
                    output_glwe_dimension,
                    output_polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            GlweDimension(output_glwe_dimension).to_glwe_size(),
            PolynomialSize(output_polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            CompressionSeed { seed },
            CiphertextModulus::new_native(),
        );
        match parallelism {
            Parallelism::No => {
                decompress_seeded_lwe_bootstrap_key::<_, _, _, SoftwareRandomGenerator>(
                    &mut output_bsk,
                    &input_bsk,
                )
            }
            Parallelism::Rayon => {
                par_decompress_seeded_lwe_bootstrap_key::<_, _, _, SoftwareRandomGenerator>(
                    &mut output_bsk,
                    &input_bsk,
                )
            }
        }
    });
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn concrete_cpu_bootstrap_key_convert_u64_to_fourier_scratch(
    stack_size: *mut usize,
    stack_align: *mut usize,
    // side resources
    fft: *const Fft,
) -> ScratchStatus {
    nounwind(|| {
        if let Ok(scratch) = convert_standard_lwe_bootstrap_key_to_fourier_mem_optimized_requirement(
            (*fft).as_view(),
        ) {
            *stack_size = scratch.size_bytes();
            *stack_align = scratch.align_bytes();
            ScratchStatus::Valid
        } else {
            ScratchStatus::SizeOverflow
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_bootstrap_key_convert_u64_to_fourier(
    // bootstrap key
    standard_bsk: *const u64,
    fourier_bsk: *mut c64,
    // bootstrap parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
    // side resources
    fft: *const Fft,
    stack: *mut u8,
    stack_size: usize,
) {
    nounwind(|| {
        let standard = LweBootstrapKey::from_container(
            slice::from_raw_parts(
                standard_bsk,
                concrete_cpu_bootstrap_key_size_u64(
                    decomposition_level_count,
                    glwe_dimension,
                    polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            CiphertextModulus::new_native(),
        );

        let mut fourier = FourierLweBootstrapKey::from_container(
            slice::from_raw_parts_mut(
                fourier_bsk,
                concrete_cpu_fourier_bootstrap_key_size_u64(
                    decomposition_level_count,
                    glwe_dimension,
                    polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            LweDimension(input_lwe_dimension),
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
        );

        convert_standard_lwe_bootstrap_key_to_fourier_mem_optimized(
            &standard,
            &mut fourier,
            (*fft).as_view(),
            PodStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
        );
    })
}

#[no_mangle]
#[must_use]
pub unsafe extern "C" fn concrete_cpu_bootstrap_lwe_ciphertext_u64_scratch(
    stack_size: *mut usize,
    stack_align: *mut usize,
    // bootstrap parameters
    glwe_dimension: usize,
    polynomial_size: usize,
    // side resources
    fft: *const Fft,
) -> ScratchStatus {
    nounwind(|| {
        if let Ok(scratch) = programmable_bootstrap_lwe_ciphertext_mem_optimized_requirement::<u64>(
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            (*fft).as_view(),
        ) {
            *stack_size = scratch.size_bytes();
            *stack_align = scratch.align_bytes();
            ScratchStatus::Valid
        } else {
            ScratchStatus::SizeOverflow
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_bootstrap_lwe_ciphertext_u64(
    // ciphertexts
    ct_out: *mut u64,
    ct_in: *const u64,
    // accumulator
    accumulator: *const u64,
    // bootstrap key
    fourier_bsk: *const c64,
    // bootstrap parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
    // side resources
    fft: *const Fft,
    stack: *mut u8,
    stack_size: usize,
) {
    nounwind(|| {
        let output_lwe_dimension = glwe_dimension * polynomial_size;

        let fourier = FourierLweBootstrapKey::from_container(
            slice::from_raw_parts(
                fourier_bsk,
                concrete_cpu_fourier_bootstrap_key_size_u64(
                    decomposition_level_count,
                    glwe_dimension,
                    polynomial_size,
                    input_lwe_dimension,
                ),
            ),
            LweDimension(input_lwe_dimension),
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
        );

        let lwe_in = LweCiphertext::from_container(
            slice::from_raw_parts(ct_in, input_lwe_dimension + 1),
            CiphertextModulus::new_native(),
        );

        let mut lwe_out = LweCiphertext::from_container(
            slice::from_raw_parts_mut(ct_out, output_lwe_dimension + 1),
            CiphertextModulus::new_native(),
        );

        let accumulator = GlweCiphertext::from_container(
            slice::from_raw_parts(
                accumulator,
                concrete_cpu_glwe_ciphertext_size_u64(glwe_dimension, polynomial_size),
            ),
            PolynomialSize(polynomial_size),
            CiphertextModulus::new_native(),
        );

        programmable_bootstrap_lwe_ciphertext_mem_optimized(
            &lwe_in,
            &mut lwe_out,
            &accumulator,
            &fourier,
            (*fft).as_view(),
            PodStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_bootstrap_key_size_u64(
    decomposition_level_count: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
) -> usize {
    input_lwe_dimension
        * ggsw_ciphertext_size(
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionLevelCount(decomposition_level_count),
        )
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_fourier_bootstrap_key_size_u64(
    decomposition_level_count: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
) -> usize {
    input_lwe_dimension
        * fourier_ggsw_ciphertext_size(
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size).to_fourier_polynomial_size(),
            DecompositionLevelCount(decomposition_level_count),
        )
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_seeded_bootstrap_key_size_u64(
    decomposition_level_count: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
) -> usize {
    input_lwe_dimension
        * seeded_ggsw_ciphertext_size(
            GlweDimension(glwe_dimension).to_glwe_size(),
            PolynomialSize(polynomial_size),
            DecompositionLevelCount(decomposition_level_count),
        )
}
