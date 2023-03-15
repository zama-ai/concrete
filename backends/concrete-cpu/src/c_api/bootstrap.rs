use crate::c_api::types::{Parallelism, ScratchStatus};
use crate::implementation::fft::Fft;
use crate::implementation::types::*;
use core::slice;
use dyn_stack::DynStack;

use super::types::{Csprng, CsprngVtable};
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
    csprng: *mut Csprng,
    csprng_vtable: *const CsprngVtable,
) {
    nounwind(|| {
        let glwe_params = GlweParams {
            dimension: output_glwe_dimension,
            polynomial_size: output_polynomial_size,
        };

        let decomp_params = DecompParams {
            level: decomposition_level_count,
            base_log: decomposition_base_log,
        };

        let bsk =
            BootstrapKey::from_raw_parts(lwe_bsk, glwe_params, input_lwe_dimension, decomp_params);

        let lwe_sk = LweSecretKey::from_raw_parts(input_lwe_sk, input_lwe_dimension);
        let glwe_sk = GlweSecretKey::from_raw_parts(output_glwe_sk, glwe_params);

        match parallelism {
            Parallelism::No => bsk.fill_with_new_key(
                lwe_sk,
                glwe_sk,
                variance,
                CsprngMut::new(csprng, csprng_vtable),
            ),
            Parallelism::Rayon => bsk.fill_with_new_key_par(
                lwe_sk,
                glwe_sk,
                variance,
                CsprngMut::new(csprng, csprng_vtable),
            ),
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
        if let Ok(scratch) = (*fft).as_view().forward_scratch() {
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
    fourier_bsk: *mut f64,
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
        let glwe_params = GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        };

        let decomp_params = DecompParams {
            level: decomposition_level_count,
            base_log: decomposition_base_log,
        };

        let standard = BootstrapKey::from_raw_parts(
            standard_bsk,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
        );

        let mut fourier = BootstrapKey::from_raw_parts(
            fourier_bsk,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
        );

        fourier.fill_with_forward_fourier(
            standard,
            (*fft).as_view(),
            DynStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
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
    fft: *const crate::implementation::fft::Fft,
) -> ScratchStatus {
    nounwind(|| {
        let fft = (*fft).as_view();
        if let Ok(scratch) = BootstrapKey::bootstrap_scratch(
            GlweParams {
                dimension: glwe_dimension,
                polynomial_size,
            },
            fft,
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
    fourier_bsk: *const f64,
    // bootstrap parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    glwe_dimension: usize,
    polynomial_size: usize,
    input_lwe_dimension: usize,
    // side resources
    fft: *const crate::implementation::fft::Fft,
    stack: *mut u8,
    stack_size: usize,
) {
    nounwind(|| {
        let glwe_params = GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        };

        let decomp_params = DecompParams {
            level: decomposition_level_count,
            base_log: decomposition_base_log,
        };

        let output_lwe_dimension = glwe_dimension * polynomial_size;

        let fourier = BootstrapKey::from_raw_parts(
            fourier_bsk,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
        );

        let lwe_in = LweCiphertext::from_raw_parts(ct_in, input_lwe_dimension);

        let lwe_out = LweCiphertext::from_raw_parts(ct_out, output_lwe_dimension);

        let accumulator = GlweCiphertext::from_raw_parts(accumulator, glwe_params);
        fourier.bootstrap(
            lwe_out,
            lwe_in,
            accumulator,
            (*fft).as_view(),
            DynStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
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
    BootstrapKey::<&[u64]>::data_len(
        GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        },
        decomposition_level_count,
        input_lwe_dimension,
    )
}
