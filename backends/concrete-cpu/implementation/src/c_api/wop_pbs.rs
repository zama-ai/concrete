use concrete_csprng::generators::SoftwareRandomGenerator;
use concrete_fft::c64;
use tfhe::core_crypto::prelude::*;

use crate::c_api::bootstrap::concrete_cpu_fourier_bootstrap_key_size_u64;
use crate::c_api::keyswitch::concrete_cpu_keyswitch_key_size_u64;
use crate::c_api::types::*;
use crate::c_api::utils::nounwind;
use core::slice;
use dyn_stack::PodStack;

use super::secret_key::{
    concrete_cpu_glwe_secret_key_size_u64, concrete_cpu_lwe_secret_key_size_u64,
};

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
    csprng: *mut EncCsprng,
) {
    nounwind(|| {
        let input_key = LweSecretKey::from_container(slice::from_raw_parts(
            input_lwe_sk,
            concrete_cpu_lwe_secret_key_size_u64(input_lwe_dimension),
        ));
        let output_key = GlweSecretKey::from_container(
            slice::from_raw_parts(
                output_glwe_sk,
                concrete_cpu_glwe_secret_key_size_u64(
                    output_glwe_dimension,
                    output_polynomial_size,
                ),
            ),
            PolynomialSize(output_polynomial_size),
        );
        let mut fpksk_list = LwePrivateFunctionalPackingKeyswitchKeyList::from_container(
            slice::from_raw_parts_mut(
                lwe_pksk,
                concrete_cpu_lwe_packing_keyswitch_key_size(
                    output_glwe_dimension,
                    output_polynomial_size,
                    decomposition_level_count,
                    input_lwe_dimension,
                ) * (output_glwe_dimension + 1),
            ),
            DecompositionBaseLog(decomposition_base_log),
            DecompositionLevelCount(decomposition_level_count),
            LweDimension(input_lwe_dimension).to_lwe_size(),
            GlweDimension(output_glwe_dimension).to_glwe_size(),
            PolynomialSize(output_polynomial_size),
            CiphertextModulus::new_native(),
        );

        match parallelism {
            Parallelism::No => generate_circuit_bootstrap_lwe_pfpksk_list(
                &mut fpksk_list,
                &input_key,
                &output_key,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
            ),
            Parallelism::Rayon => par_generate_circuit_bootstrap_lwe_pfpksk_list(
                &mut fpksk_list,
                &input_key,
                &output_key,
                Gaussian::from_dispersion_parameter(Variance::from_variance(variance), 0.0),
                &mut *(csprng as *mut EncryptionRandomGenerator<SoftwareRandomGenerator>),
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
        if let Ok(scratch) = extract_bits_from_lwe_ciphertext_mem_optimized_requirement::<u64>(
            LweDimension(ct_in_dimension),
            LweDimension(ct_out_dimension + 1),
            GlweDimension(bsk_glwe_dimension).to_glwe_size(),
            PolynomialSize(bsk_polynomial_size),
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
    fourier_bsk: *const c64,
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

        let mut lwe_list_out = LweCiphertextList::from_container(
            slice::from_raw_parts_mut(ct_vec_out, (ct_out_dimension + 1) * ct_out_count),
            LweDimension(ct_out_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        let lwe_in = LweCiphertext::from_container(
            slice::from_raw_parts(ct_in, ct_in_dimension + 1),
            CiphertextModulus::new_native(),
        );

        let ksk = LweKeyswitchKey::from_container(
            slice::from_raw_parts(
                ksk,
                concrete_cpu_keyswitch_key_size_u64(
                    ksk_decomposition_level_count,
                    ksk_input_dimension,
                    ksk_output_dimension,
                ),
            ),
            DecompositionBaseLog(ksk_decomposition_base_log),
            DecompositionLevelCount(ksk_decomposition_level_count),
            LweDimension(ksk_output_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        let fourier_bsk = FourierLweBootstrapKey::from_container(
            slice::from_raw_parts(
                fourier_bsk,
                concrete_cpu_fourier_bootstrap_key_size_u64(
                    bsk_decomposition_level_count,
                    bsk_glwe_dimension,
                    bsk_polynomial_size,
                    bsk_input_lwe_dimension,
                ),
            ),
            LweDimension(bsk_input_lwe_dimension),
            GlweDimension(bsk_glwe_dimension).to_glwe_size(),
            PolynomialSize(bsk_polynomial_size),
            DecompositionBaseLog(bsk_decomposition_base_log),
            DecompositionLevelCount(bsk_decomposition_level_count),
        );

        extract_bits_from_lwe_ciphertext_mem_optimized(
            &lwe_in,
            &mut lwe_list_out,
            &fourier_bsk,
            &ksk,
            DeltaLog(delta_log),
            ExtractedBitsCount(number_of_bits),
            (*fft).as_view(),
            PodStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
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

        if let Ok(scratch) =
            circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_list_mem_optimized_requirement::<
                u64,
            >(
                LweCiphertextCount(ct_in_count),
                LweCiphertextCount(ct_out_count),
                LweDimension(ct_in_dimension).to_lwe_size(),
                PolynomialCount(lut_count),
                LweDimension(bsk_output_lwe_dimension).to_lwe_size(),
                GlweDimension(bsk_glwe_dimension).to_glwe_size(),
                PolynomialSize(fpksk_output_polynomial_size.max(lut_size)),
                DecompositionLevelCount(cbs_decomposition_level_count),
                (*fft).as_view(),
            )
        {
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
    fourier_bsk: *const c64,
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
    _fpksk_count: usize,
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

        let mut lut_container = slice::from_raw_parts(lut, lut_size * lut_count);
        let mut expanded_luts: Vec<u64> = vec![0_u64; fpksk_output_polynomial_size * lut_count];

        if lut_size < fpksk_output_polynomial_size {
            for luti in 0..lut_count {
                for i in 0..lut_size {
                    expanded_luts[luti * fpksk_output_polynomial_size + i] =
                        lut_container[luti * lut_size + i];
                }
            }
            lut_container = expanded_luts.as_slice();
        }

        let luts = PolynomialList::from_container(
            lut_container,
            PolynomialSize(fpksk_output_polynomial_size),
        );

        let fourier_bsk = FourierLweBootstrapKey::from_container(
            slice::from_raw_parts(
                fourier_bsk,
                concrete_cpu_fourier_bootstrap_key_size_u64(
                    bsk_decomposition_level_count,
                    bsk_glwe_dimension,
                    bsk_polynomial_size,
                    bsk_input_lwe_dimension,
                ),
            ),
            LweDimension(bsk_input_lwe_dimension),
            GlweDimension(bsk_glwe_dimension).to_glwe_size(),
            PolynomialSize(bsk_polynomial_size),
            DecompositionBaseLog(bsk_decomposition_base_log),
            DecompositionLevelCount(bsk_decomposition_level_count),
        );

        let mut lwe_list_out = LweCiphertextList::from_container(
            slice::from_raw_parts_mut(ct_out_vec, (ct_out_dimension + 1) * ct_out_count),
            LweDimension(ct_out_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        let lwe_list_in = LweCiphertextList::from_container(
            slice::from_raw_parts(ct_in_vec, (ct_in_dimension + 1) * ct_in_count),
            LweDimension(ct_in_dimension).to_lwe_size(),
            CiphertextModulus::new_native(),
        );

        let fpksk_list = LwePrivateFunctionalPackingKeyswitchKeyList::from_container(
            slice::from_raw_parts(
                fpksk,
                concrete_cpu_lwe_packing_keyswitch_key_size(
                    fpksk_output_glwe_dimension,
                    fpksk_output_polynomial_size,
                    fpksk_decomposition_level_count,
                    fpksk_input_dimension,
                ) * (fpksk_output_glwe_dimension + 1),
            ),
            DecompositionBaseLog(fpksk_decomposition_base_log),
            DecompositionLevelCount(fpksk_decomposition_level_count),
            LweDimension(fpksk_input_dimension).to_lwe_size(),
            GlweDimension(fpksk_output_glwe_dimension).to_glwe_size(),
            PolynomialSize(fpksk_output_polynomial_size),
            CiphertextModulus::new_native(),
        );

        circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_list_mem_optimized(
            &lwe_list_in,
            &mut lwe_list_out,
            &luts,
            &fourier_bsk,
            &fpksk_list,
            DecompositionBaseLog(cbs_decomposition_base_log),
            DecompositionLevelCount(cbs_decomposition_level_count),
            (*fft).as_view(),
            PodStack::new(slice::from_raw_parts_mut(stack as _, stack_size)),
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_lwe_packing_keyswitch_key_size(
    output_glwe_dimension: usize,
    polynomial_size: usize,
    decomposition_level_count: usize,
    input_lwe_dimension: usize,
) -> usize {
    lwe_pfpksk_size(
        LweDimension(input_lwe_dimension).to_lwe_size(),
        DecompositionLevelCount(decomposition_level_count),
        GlweDimension(output_glwe_dimension).to_glwe_size(),
        PolynomialSize(polynomial_size),
    )
}
