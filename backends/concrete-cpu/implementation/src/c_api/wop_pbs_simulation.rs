use crate::c_api::utils::nounwind;
use crate::implementation::wop_simulation::{
    circuit_bootstrap_boolean_vertical_packing, extract_bits,
};
use concrete_csprng::generators::SoftwareRandomGenerator;
use core::slice;
use tfhe::core_crypto::commons::math::random::RandomGenerator;

use super::types::Csprng;

#[no_mangle]
pub unsafe extern "C" fn simulation_extract_bit_lwe_ciphertext_u64(
    lwe_list_out: *mut u64,
    lwe_in: u64,
    delta_log: usize,
    number_of_bits_to_extract: usize,
    log_poly_size: u64,
    glwe_dimension: u64,
    lwe_dimension: u64,
    ks_log_base: u64,
    ks_level: u64,
    br_log_base: u64,
    br_level: u64,
    ciphertext_modulus_log: u32,
    security_level: u64,
    csprng: *mut Csprng,
) {
    nounwind(|| {
        assert!(64 <= number_of_bits_to_extract + delta_log);

        let csprng = &mut *(csprng as *mut RandomGenerator<SoftwareRandomGenerator>);

        extract_bits(
            slice::from_raw_parts_mut(lwe_list_out, number_of_bits_to_extract),
            lwe_in,
            delta_log,
            number_of_bits_to_extract,
            log_poly_size,
            glwe_dimension,
            lwe_dimension,
            ks_log_base,
            ks_level,
            br_log_base,
            br_level,
            ciphertext_modulus_log,
            security_level,
            csprng,
        );
    })
}

#[no_mangle]
pub unsafe extern "C" fn simulation_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_u64(
    lwe_list_in: *const u64,
    lwe_list_out: *mut u64,
    ct_in_count: usize,
    ct_out_count: usize,
    lut_size: usize,
    lut_count: usize,
    luts: *const u64,
    glwe_dimension: u64,
    log_poly_size: u64,
    lwe_dimension: u64,
    pbs_level: u64,
    pbs_log_base: u64,
    cb_level: u64,
    cb_log_base: u64,
    pp_level: u64,
    pp_log_base: u64,
    ciphertext_modulus_log: u32,
    security_level: u64,
    csprng: *mut Csprng,
) {
    nounwind(|| {
        assert_ne!(cb_log_base, 0);
        assert_ne!(cb_level, 0);
        assert!(cb_level * cb_log_base <= 64);

        let luts = slice::from_raw_parts(luts, lut_count * lut_size);

        let lwe_list_out = slice::from_raw_parts_mut(lwe_list_out, ct_out_count);

        let lwe_list_in = slice::from_raw_parts(lwe_list_in, ct_in_count);

        let csprng = &mut *(csprng as *mut RandomGenerator<SoftwareRandomGenerator>);

        circuit_bootstrap_boolean_vertical_packing(
            lwe_list_in,
            lwe_list_out,
            luts,
            glwe_dimension,
            log_poly_size,
            lwe_dimension,
            pbs_level,
            pbs_log_base,
            cb_level,
            cb_log_base,
            pp_level,
            pp_log_base,
            ciphertext_modulus_log,
            security_level,
            csprng,
        );
    })
}
