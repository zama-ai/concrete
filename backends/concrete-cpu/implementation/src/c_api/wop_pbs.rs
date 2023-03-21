use crate::c_api::types::*;
use crate::c_api::utils::nounwind;
use crate::implementation::fft::Fft;
use crate::implementation::types::ciphertext_list::LweCiphertextList;
use crate::implementation::types::packing_keyswitch_key_list::PackingKeyswitchKeyList;
use crate::implementation::types::polynomial_list::PolynomialList;
use crate::implementation::types::*;
use crate::implementation::wop::{
    circuit_bootstrap_boolean_vertical_packing, circuit_bootstrap_boolean_vertical_packing_scratch,
    extract_bits, extract_bits_scratch,
};
use core::slice;
use dyn_stack::DynStack;

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
    // packing keyswitch key
    lwe_pksk: *mut u64,
    // secret keys
    input_lwe_sk: *const u64,
    output_glwe_sk: *const u64,
    // secret key dimensions
    input_lwe_dimension: usize,
    output_polynomial_size: usize,
    output_glwe_dimension: usize,
    // circuit bootstrap parameters
    decomposition_level_count: usize,
    decomposition_base_log: usize,
    // noise parameters
    variance: f64,
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

        let input_key = LweSecretKey::<&[u64]>::from_raw_parts(input_lwe_sk, input_lwe_dimension);
        let output_key = GlweSecretKey::<&[u64]>::from_raw_parts(output_glwe_sk, glwe_params);
        let mut fpksk_list = PackingKeyswitchKeyList::<&mut [u64]>::from_raw_parts(
            lwe_pksk,
            glwe_params,
            input_lwe_dimension,
            decomp_params,
            glwe_params.dimension + 1,
        );

        match parallelism {
            Parallelism::No => fpksk_list.fill_with_fpksk_for_circuit_bootstrap(
                &input_key,
                &output_key,
                variance,
                CsprngMut::new(csprng, csprng_vtable),
            ),
            Parallelism::Rayon => fpksk_list.fill_with_fpksk_for_circuit_bootstrap_par(
                &input_key,
                &output_key,
                variance,
                CsprngMut::new(csprng, csprng_vtable),
            ),
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_extract_bit_lwe_ciphertext_u64_scratch(
    stack_size: *mut usize,
    stack_align: *mut usize,
    // ciphertexts dimensions
    ct_out_dimension: usize,
    ct_in_dimension: usize,
    // bootstrap parameters
    bsk_glwe_dimension: usize,
    bsk_polynomial_size: usize,
    // side resources
    fft: *const Fft,
) -> ScratchStatus {
    nounwind(|| {
        if let Ok(scratch) = extract_bits_scratch(
            ct_in_dimension,
            ct_out_dimension + 1,
            GlweParams {
                dimension: bsk_glwe_dimension,
                polynomial_size: bsk_polynomial_size,
            },
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
pub unsafe extern "C" fn concrete_cpu_extract_bit_lwe_ciphertext_u64(
    // ciphertexts
    ct_vec_out: *mut u64,
    ct_in: *const u64,
    // bootstrap key
    fourier_bsk: *const f64,
    // keyswitch key
    ksk: *const u64,
    // ciphertexts dimensions
    ct_out_dimension: usize,
    ct_out_count: usize,
    ct_in_dimension: usize,
    // extract bit parameters
    number_of_bits: usize,
    delta_log: usize,
    // bootstrap parameters
    bsk_decomposition_level_count: usize,
    bsk_decomposition_base_log: usize,
    bsk_glwe_dimension: usize,
    bsk_polynomial_size: usize,
    bsk_input_lwe_dimension: usize,
    // keyswitch_parameters
    ksk_decomposition_level_count: usize,
    ksk_decomposition_base_log: usize,
    ksk_input_dimension: usize,
    ksk_output_dimension: usize,
    // side resources
    fft: *const Fft,
    stack: *mut u8,
    stack_size: usize,
) {
    nounwind(|| {
        assert_eq!(ct_in_dimension, bsk_glwe_dimension * bsk_polynomial_size);
        assert_eq!(ct_in_dimension, ksk_input_dimension);
        assert_eq!(ct_out_dimension, ksk_output_dimension);
        assert_eq!(ct_out_count, number_of_bits);
        assert_eq!(ksk_output_dimension, bsk_input_lwe_dimension);
        assert!(64 <= number_of_bits + delta_log);

        let lwe_list_out =
            LweCiphertextList::from_raw_parts(ct_vec_out, ct_out_dimension, ct_out_count);

        let lwe_in = LweCiphertext::from_raw_parts(ct_in, ct_in_dimension);

        let ksk = LweKeyswitchKey::from_raw_parts(
            ksk,
            ksk_output_dimension,
            ksk_input_dimension,
            DecompParams {
                level: ksk_decomposition_level_count,
                base_log: ksk_decomposition_base_log,
            },
        );

        let fourier_bsk = BootstrapKey::from_raw_parts(
            fourier_bsk,
            GlweParams {
                dimension: bsk_glwe_dimension,
                polynomial_size: bsk_polynomial_size,
            },
            bsk_input_lwe_dimension,
            DecompParams {
                level: bsk_decomposition_level_count,
                base_log: bsk_decomposition_base_log,
            },
        );

        extract_bits(
            lwe_list_out,
            lwe_in,
            ksk,
            fourier_bsk,
            delta_log,
            number_of_bits,
            (*fft).as_view(),
            DynStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64_scratch(
    stack_size: *mut usize,
    stack_align: *mut usize,
    // ciphertext dimensions
    ct_out_count: usize,
    ct_in_dimension: usize,
    ct_in_count: usize,
    lut_size: usize,
    lut_count: usize,
    // bootstrap parameters
    bsk_glwe_dimension: usize,
    bsk_polynomial_size: usize,
    // keyswitch_parameters
    fpksk_output_polynomial_size: usize,
    // circuit bootstrap parameters
    cbs_decomposition_level_count: usize,
    // side resources
    fft: *const Fft,
) -> ScratchStatus {
    nounwind(|| {
        assert_eq!(ct_out_count, lut_count);
        let bsk_output_lwe_dimension = bsk_glwe_dimension * bsk_polynomial_size;

        assert_eq!(lut_size, 1 << ct_in_count);

        assert_ne!(cbs_decomposition_level_count, 0);

        if let Ok(scratch) = circuit_bootstrap_boolean_vertical_packing_scratch(
            ct_in_count,
            ct_out_count,
            ct_in_dimension + 1,
            lut_count,
            bsk_output_lwe_dimension + 1,
            fpksk_output_polynomial_size,
            bsk_glwe_dimension + 1,
            cbs_decomposition_level_count,
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
pub unsafe extern "C" fn concrete_cpu_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
    // ciphertexts
    ct_out_vec: *mut u64,
    ct_in_vec: *const u64,
    // lookup table
    lut: *const u64,
    // bootstrap key
    fourier_bsk: *const f64,
    // packing keyswitch key
    fpksk: *const u64,
    // ciphertext dimensions
    ct_out_dimension: usize,
    ct_out_count: usize,
    ct_in_dimension: usize,
    ct_in_count: usize,
    lut_size: usize,
    lut_count: usize,
    // bootstrap parameters
    bsk_decomposition_level_count: usize,
    bsk_decomposition_base_log: usize,
    bsk_glwe_dimension: usize,
    bsk_polynomial_size: usize,
    bsk_input_lwe_dimension: usize,
    // keyswitch_parameters
    fpksk_decomposition_level_count: usize,
    fpksk_decomposition_base_log: usize,
    fpksk_input_dimension: usize,
    fpksk_output_glwe_dimension: usize,
    fpksk_output_polynomial_size: usize,
    fpksk_count: usize,
    // circuit bootstrap parameters
    cbs_decomposition_level_count: usize,
    cbs_decomposition_base_log: usize,
    // side resources
    fft: *const Fft,
    stack: *mut u8,
    stack_size: usize,
) {
    nounwind(|| {
        assert_eq!(ct_out_count, lut_count);
        let bsk_output_lwe_dimension = bsk_glwe_dimension * bsk_polynomial_size;
        assert_eq!(bsk_output_lwe_dimension, fpksk_input_dimension);
        assert_eq!(ct_in_dimension, bsk_input_lwe_dimension);
        assert_eq!(
            ct_out_dimension,
            fpksk_output_glwe_dimension * fpksk_output_polynomial_size
        );
        assert_eq!(lut_size, 1 << ct_in_count);

        assert_ne!(cbs_decomposition_base_log, 0);
        assert_ne!(cbs_decomposition_level_count, 0);
        assert!(cbs_decomposition_level_count * cbs_decomposition_base_log <= 64);

        let bsk_glwe_params = GlweParams {
            dimension: bsk_glwe_dimension,
            polynomial_size: bsk_polynomial_size,
        };

        let luts = PolynomialList::new(
            slice::from_raw_parts(lut, lut_size * lut_count),
            lut_size,
            lut_count,
        );

        let fourier_bsk = BootstrapKey::<&[f64]>::from_raw_parts(
            fourier_bsk,
            bsk_glwe_params,
            bsk_input_lwe_dimension,
            DecompParams {
                level: bsk_decomposition_level_count,
                base_log: bsk_decomposition_base_log,
            },
        );

        let lwe_list_out = LweCiphertextList::<&mut [u64]>::from_raw_parts(
            ct_out_vec,
            ct_out_dimension,
            ct_out_count,
        );

        let lwe_list_in =
            LweCiphertextList::<&[u64]>::from_raw_parts(ct_in_vec, ct_in_dimension, ct_in_count);

        let fpksk_list = PackingKeyswitchKeyList::new(
            slice::from_raw_parts(
                fpksk,
                fpksk_decomposition_level_count
                    * (fpksk_output_glwe_dimension + 1)
                    * fpksk_output_polynomial_size
                    * (fpksk_input_dimension + 1)
                    * fpksk_count,
            ),
            GlweParams {
                dimension: fpksk_output_glwe_dimension,
                polynomial_size: fpksk_output_polynomial_size,
            },
            fpksk_input_dimension,
            DecompParams {
                level: fpksk_decomposition_level_count,
                base_log: fpksk_decomposition_base_log,
            },
            fpksk_count,
        );

        circuit_bootstrap_boolean_vertical_packing(
            luts,
            fourier_bsk,
            lwe_list_out,
            lwe_list_in,
            fpksk_list,
            DecompParams {
                level: cbs_decomposition_level_count,
                base_log: cbs_decomposition_base_log,
            },
            (*fft).as_view(),
            DynStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_packing_keyswitch_key_size(
    glwe_dimension: usize,
    polynomial_size: usize,
    decomposition_level_count: usize,
    input_dimension: usize,
) -> usize {
    PackingKeyswitchKey::<&[u64]>::data_len(
        GlweParams {
            dimension: glwe_dimension,
            polynomial_size,
        },
        decomposition_level_count,
        input_dimension,
    )
}
