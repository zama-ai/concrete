use super::Slice;
pub use concrete_commons::dispersion::Variance;
use concrete_commons::key_kinds::{BinaryKeyKind, GaussianKeyKind, TernaryKeyKind, UniformKeyKind};
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_npe::*;
use std::slice::from_raw_parts;

#[no_mangle]
pub extern "C" fn variance_add_u32(dispersion_ct1: Variance, dispersion_ct2: Variance) -> Variance {
    return variance_add::<u32, Variance, Variance>(dispersion_ct1, dispersion_ct2);
}
#[no_mangle]
pub extern "C" fn variance_add_u64(dispersion_ct1: Variance, dispersion_ct2: Variance) -> Variance {
    return variance_add::<u64, Variance, Variance>(dispersion_ct1, dispersion_ct2);
}
#[no_mangle]
pub extern "C" fn variance_add_several_u32(dispersion_cts: VarianceArray) -> Variance {
    unsafe {
        return variance_add_several::<u32, Variance>(from_raw_parts(
            dispersion_cts.ptr.as_ptr() as *const Variance,
            dispersion_cts.len,
        ));
    }
}
#[no_mangle]
pub extern "C" fn variance_add_several_u64(dispersion_cts: VarianceArray) -> Variance {
    unsafe {
        return variance_add_several::<u64, Variance>(from_raw_parts(
            dispersion_cts.ptr.as_ptr() as *const Variance,
            dispersion_cts.len,
        ));
    }
}
#[no_mangle]
pub extern "C" fn variance_external_product_u32_binary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u32, Variance, Variance, BinaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_external_product_u64_binary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u64, Variance, Variance, BinaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_external_product_u32_ternary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u32, Variance, Variance, TernaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_external_product_u64_ternary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u64, Variance, Variance, TernaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_external_product_u32_gaussian_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u32, Variance, Variance, GaussianKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_external_product_u64_gaussian_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_ggsw: Variance,
    dispersion_glwe: Variance,
) -> Variance {
    return variance_external_product::<u64, Variance, Variance, GaussianKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_ggsw,
        dispersion_glwe,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u32_binary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u32, Variance, Variance, Variance, BinaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u64_binary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u64, Variance, Variance, Variance, BinaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u32_ternary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u32, Variance, Variance, Variance, TernaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u64_ternary_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u64, Variance, Variance, Variance, TernaryKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u32_gaussian_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u32, Variance, Variance, Variance, GaussianKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_cmux_u64_gaussian_key(
    dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    base_log: DecompositionBaseLog,
    l_gadget: DecompositionLevelCount,
    dispersion_rlwe_0: Variance,
    dispersion_rlwe_1: Variance,
    dispersion_rgsw: Variance,
) -> Variance {
    return variance_cmux::<u64, Variance, Variance, Variance, GaussianKeyKind>(
        dimension,
        polynomial_size,
        base_log,
        l_gadget,
        dispersion_rlwe_0,
        dispersion_rlwe_1,
        dispersion_rgsw,
    );
}
#[no_mangle]
pub extern "C" fn variance_scalar_mul_u32(variance: Variance, n: u32) -> Variance {
    return variance_scalar_mul::<u32, Variance>(variance, n);
}
#[no_mangle]
pub extern "C" fn variance_scalar_mul_u64(variance: Variance, n: u64) -> Variance {
    return variance_scalar_mul::<u64, Variance>(variance, n);
}
#[no_mangle]
pub extern "C" fn variance_scalar_weighted_sum_u32(
    dispersion_list: VarianceArray,
    weights: u32Array,
) -> Variance {
    unsafe {
        return variance_scalar_weighted_sum::<u32, Variance>(
            from_raw_parts(dispersion_list.ptr.as_ptr() as *const Variance, dispersion_list.len),
            from_raw_parts(weights.ptr.as_ptr() as *const u32, weights.len));
    }
}
#[no_mangle]
pub extern "C" fn variance_scalar_weighted_sum_u64(
    dispersion_list: VarianceArray,
    weights: u64Array,
) -> Variance {
    unsafe {
        return variance_scalar_weighted_sum::<u64, Variance>(
            from_raw_parts(dispersion_list.ptr.as_ptr() as *const Variance, dispersion_list.len),
            from_raw_parts(weights.ptr.as_ptr() as * u64, weights.len));
    }
}
#[no_mangle]
pub extern "C" fn variance_scalar_polynomial_mul_u32(
    dispersion: Variance,
    scalar_polynomial: u32Array,
) -> Variance {
    unsafe {
        return variance_scalar_polynomial_mul::<u32, Variance>(dispersion, from_raw_parts
            (scalar_polynomial.ptr.as_ptr() as *const u32, scalar_polynomial.len));
    }
}
#[no_mangle]
pub extern "C" fn variance_scalar_polynomial_mul_u64(
    dispersion: Variance,
    scalar_polynomial: u64Array,
) -> Variance {
        unsafe {
            return variance_scalar_polynomial_mul::<u64, Variance>(dispersion, from_raw_parts
                (scalar_polynomial.ptr.as_ptr() as *const u64, scalar_polynomial.len));
        }
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u32_binary_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u32, Variance, Variance, BinaryKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u64_binary_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u64, Variance, Variance, BinaryKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u32_ternary_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u32, Variance, Variance, TernaryKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u64_ternary_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u64, Variance, Variance, TernaryKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u32_gaussian_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u32, Variance, Variance, GaussianKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_tensor_product_rescale_round_u64_gaussian_key(
    poly_size: PolynomialSize,
    rlwe_dimension: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
) -> Variance {
    return variance_glwe_tensor_product_rescale_round::<u64, Variance, Variance, GaussianKeyKind>(
        poly_size,
        rlwe_dimension,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u32_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u32, Variance, BinaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u64_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u64, Variance, BinaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u32_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u32, Variance, TernaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u64_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u64, Variance, TernaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u32_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u32, Variance, GaussianKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_relinearization_u64_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_relinearization::<u64, Variance, GaussianKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u32_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u32,
        Variance,
        Variance,
        Variance,
        BinaryKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u64_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u64,
        Variance,
        Variance,
        Variance,
        BinaryKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u32_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u32,
        Variance,
        Variance,
        Variance,
        TernaryKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u64_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u64,
        Variance,
        Variance,
        Variance,
        TernaryKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u32_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u32,
        Variance,
        Variance,
        Variance,
        GaussianKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_glwe_mul_with_relinearization_u64_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_glwe1: Variance,
    dispersion_glwe2: Variance,
    delta_1: f64,
    delta_2: f64,
    max_msg_1: f64,
    max_msg_2: f64,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_glwe_mul_with_relinearization::<
        u64,
        Variance,
        Variance,
        Variance,
        GaussianKeyKind,
    >(
        poly_size,
        mask_size,
        dispersion_glwe1,
        dispersion_glwe2,
        delta_1,
        delta_2,
        max_msg_1,
        max_msg_2,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_lwe_drift_pbs_with_binary_key_u32(
    lwe_mask_size: LweDimension,
    nb_msb: usize,
    var_in: Variance,
) -> Variance {
    return variance_lwe_drift_pbs_with_binary_key::<u32, Variance>(lwe_mask_size, nb_msb, var_in);
}
#[no_mangle]
pub extern "C" fn variance_lwe_drift_pbs_with_binary_key_u64(
    lwe_mask_size: LweDimension,
    nb_msb: usize,
    var_in: Variance,
) -> Variance {
    return variance_lwe_drift_pbs_with_binary_key::<u64, Variance>(lwe_mask_size, nb_msb, var_in);
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u32_binary_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u32, Variance, Variance, BinaryKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u64_binary_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u64, Variance, Variance, BinaryKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u32_ternary_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u32, Variance, Variance, TernaryKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u64_ternary_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u64, Variance, Variance, TernaryKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u32_gaussian_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u32, Variance, Variance, GaussianKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_constant_term_u64_gaussian_key(
    lwe_mask_size: LweDimension,
    dispersion_lwe: Variance,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_constant_term::<u64, Variance, Variance, GaussianKeyKind>(
        lwe_mask_size,
        dispersion_lwe,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_non_constant_terms_u32(
    lwe_mask_size: LweDimension,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_non_constant_terms::<u32, Variance>(
        lwe_mask_size,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_keyswitch_lwe_to_glwe_non_constant_terms_u64(
    lwe_mask_size: LweDimension,
    dispersion_ksk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_keyswitch_lwe_to_glwe_non_constant_terms::<u64, Variance>(
        lwe_mask_size,
        dispersion_ksk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u32_binary_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u32, BinaryKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u64_binary_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u64, BinaryKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u32_ternary_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u32, TernaryKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u64_ternary_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u64, TernaryKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u32_gaussian_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u32, GaussianKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_u_rlwe_k_1_mod_switch_u64_gaussian_key(
    poly_size: PolynomialSize,
) -> Variance {
    return variance_u_rlwe_k_1_mod_switch::<u64, GaussianKeyKind>(poly_size);
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u32_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u32, Variance, BinaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u64_binary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u64, Variance, BinaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u32_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u32, Variance, TernaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u64_ternary_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u64, Variance, TernaryKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u32_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u32, Variance, GaussianKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_rlwe_relinearization_u64_gaussian_key(
    poly_size: PolynomialSize,
    mask_size: GlweDimension,
    dispersion_rlk: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_rlwe_relinearization::<u64, Variance, GaussianKeyKind>(
        poly_size,
        mask_size,
        dispersion_rlk,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u32_binary_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u32, Variance, Variance, BinaryKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u64_binary_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u64, Variance, Variance, BinaryKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u32_ternary_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u32, Variance, Variance, TernaryKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u64_ternary_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u64, Variance, Variance, TernaryKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u32_gaussian_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u32, Variance, Variance, GaussianKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub fn variance_external_product_binary_ggsw_u64_gaussian_key(
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_glwe: Variance,
    var_ggsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_external_product_binary_ggsw::<u64, Variance, Variance, GaussianKeyKind>(
        poly_size,
        rlwe_mask_size,
        var_glwe,
        var_ggsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u32_binary_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u32, Variance, BinaryKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u64_binary_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u64, Variance, BinaryKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u32_ternary_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u32, Variance, TernaryKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u64_ternary_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u64, Variance, TernaryKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u32_gaussian_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u32, Variance, GaussianKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
#[no_mangle]
pub extern "C" fn variance_tfhe_pbs_u64_gaussian_key(
    lwe_mask_size: LweDimension,
    poly_size: PolynomialSize,
    rlwe_mask_size: GlweDimension,
    var_rgsw: Variance,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
) -> Variance {
    return variance_tfhe_pbs::<u64, Variance, GaussianKeyKind>(
        lwe_mask_size,
        poly_size,
        rlwe_mask_size,
        var_rgsw,
        base_log,
        level,
    );
}
