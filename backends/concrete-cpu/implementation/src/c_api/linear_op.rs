use super::utils::nounwind;
use core::slice;

/// # Safety
///
/// `[ct_out, ct_out + lwe_dimension + 1[` must be a valid mutable range, and must not alias
/// `[ct_in0, ct_in0 + lwe_dimension + 1[` or `[ct_in1, ct_in1 + lwe_dimension + 1[`, both of which
/// must be valid ranges for reads.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_add_lwe_ciphertext_u64(
    ct_out: *mut u64,
    ct_in0: *const u64,
    ct_in1: *const u64,
    lwe_dimension: usize,
) {
    nounwind(|| {
        #[inline]
        fn implementation(ct_out: &mut [u64], ct_in0: &[u64], ct_in1: &[u64]) {
            for ((out, &c0), &c1) in ct_out.iter_mut().zip(ct_in0).zip(ct_in1) {
                *out = c0.wrapping_add(c1)
            }
        }

        let lwe_size = lwe_dimension + 1;
        pulp::Arch::new().dispatch(|| {
            implementation(
                slice::from_raw_parts_mut(ct_out, lwe_size),
                slice::from_raw_parts(ct_in0, lwe_size),
                slice::from_raw_parts(ct_in1, lwe_size),
            )
        });
    })
}

/// # Safety
///
/// `[ct_out, ct_out + lwe_dimension + 1[` must be a valid mutable range, and must not alias
/// `[ct_in, ct_in + lwe_dimension + 1[`, which must be a valid range for reads.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_add_plaintext_lwe_ciphertext_u64(
    ct_out: *mut u64,
    ct_in: *const u64,
    plaintext: u64,
    lwe_dimension: usize,
) {
    nounwind(|| {
        #[inline]
        fn implementation(ct_out: &mut [u64], ct_in: &[u64], plaintext: u64) {
            ct_out.copy_from_slice(ct_in);

            let last = ct_out.last_mut().unwrap();

            *last = last.wrapping_add(plaintext);
        }

        let lwe_size = lwe_dimension + 1;
        pulp::Arch::new().dispatch(|| {
            implementation(
                slice::from_raw_parts_mut(ct_out, lwe_size),
                slice::from_raw_parts(ct_in, lwe_size),
                plaintext,
            )
        });
    })
}

/// # Safety
///
/// `[ct_out, ct_out + lwe_dimension + 1[` must be a valid mutable range, and must not alias
/// `[ct_in, ct_in + lwe_dimension + 1[`, which must be a valid range for reads.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_mul_cleartext_lwe_ciphertext_u64(
    ct_out: *mut u64,
    ct_in: *const u64,
    cleartext: u64,
    lwe_dimension: usize,
) {
    nounwind(|| {
        #[inline]
        fn implementation(ct_out: &mut [u64], ct_in: &[u64], cleartext: u64) {
            for (out, &c) in ct_out.iter_mut().zip(ct_in) {
                *out = c.wrapping_mul(cleartext)
            }
        }

        let lwe_size = lwe_dimension + 1;
        pulp::Arch::new().dispatch(|| {
            implementation(
                slice::from_raw_parts_mut(ct_out, lwe_size),
                slice::from_raw_parts(ct_in, lwe_size),
                cleartext,
            )
        });
    })
}

/// # Safety
///
/// `[ct_out, ct_out + lwe_dimension + 1[` must be a valid mutable range, and must not alias
/// `[ct_in, ct_in + lwe_dimension + 1[`, which must be a valid range for reads.
#[no_mangle]
pub unsafe extern "C" fn concrete_cpu_negate_lwe_ciphertext_u64(
    ct_out: *mut u64,
    ct_in: *const u64,
    lwe_dimension: usize,
) {
    nounwind(|| {
        #[inline]
        fn implementation(ct_out: &mut [u64], ct_in: &[u64]) {
            for (out, &c) in ct_out.iter_mut().zip(ct_in) {
                *out = c.wrapping_neg();
            }
        }

        let lwe_size = lwe_dimension + 1;

        pulp::Arch::new().dispatch(|| {
            implementation(
                slice::from_raw_parts_mut(ct_out, lwe_size),
                slice::from_raw_parts(ct_in, lwe_size),
            )
        });
    })
}
