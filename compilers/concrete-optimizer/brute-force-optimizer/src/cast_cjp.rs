use crate::casting_keyswitch::{CastingConstraint, CastingSearchSpace};
use crate::cjp::{CJPConstraint, CJPSearchSpace};
use crate::generic::SequentialProblem;
use crate::MyRange;
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_optimizer::noise_estimator::error;
use concrete_security_curves::gaussian::security::minimal_variance_glwe;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::io::Write;
use std::time::Instant;

pub fn norm(precision_block: u64, carry: u64) -> u64 {
    ((f64::exp2((precision_block + carry) as f64) - 1.) / (f64::exp2(carry as f64) - 1.)).floor()
        as u64
}

pub fn solve_all_cast_cjp(p_fail: f64, writer: impl Write) {
    let start = Instant::now();
    let TOTAL_PRECISION_CARRY = 8;
    let mut experiments: Vec<(u64, u64)> = vec![(1u64, 1), (2, 2)];
    // vec![(2u64, 2), (1, 1)];

    let mut experiments_with_2norms = {
        let first_partition = experiments[0];
        let second_partition = experiments[1];
        let (first_pblock, first_carry) = first_partition;
        let (second_pblock, second_carry) = second_partition;
        vec![
            (
                first_pblock + first_carry,
                first_carry,
                norm(first_pblock, first_carry),
            ),
            (
                second_pblock + second_carry,
                second_carry,
                norm(second_pblock, second_carry),
            ),
        ]
    };

    let a = CJPSearchSpace {
        range_base_log_ks: MyRange(1, 40),
        range_level_ks: MyRange(1, 40),
        _range_base_log_pbs: MyRange(1, 40),
        _range_level_pbs: MyRange(1, 53),
        range_glwe_dim: MyRange(1, 7),
        range_log_poly_size: MyRange(8, 19),
        range_small_lwe_dim: MyRange(500, 1500),
    };
    let minimal_ms_value = 0;

    let a_tighten = a.to_tighten(128);
    let res: Vec<_> = experiments_with_2norms
        .iter()
        .map(|(precision, carry, norm)| {
            let norm = if *precision == 2 { *norm } else { *norm };
            let config = CJPConstraint {
                variance_constraint: error::safe_variance_bound_2padbits(*precision, 64, p_fail), //5.960464477539063e-08, // 0.0009765625006088146,
                norm2: norm,
                security_level: 128,
                sum_size: 4096,
            };

            let interm = config.brute_force(a_tighten.clone().iter(*precision, minimal_ms_value));

            (precision, carry, interm)
        })
        .collect::<Vec<_>>();

    // we have the parameters for (1, 1) and for (2, 2)
    // now we want a keyswitch to do (1, 1) -> (2,2)
    // var_bind_rotate_1 * norm2^2 + var_new_ks + var_ms_2 < noise_bound(2)
    let params_1 = match res[0] {
        (_, _, Some((param, _))) => param,
        _ => panic!(),
    };
    let params_2 = match res[1] {
        (_, _, Some((param, _))) => param,
        _ => panic!(),
    };
    let variance_bsk =
        minimal_variance_glwe(params_1.glwe_dim, 1 << params_1.log_poly_size, 64, 128);
    let v_pbs = variance_blind_rotate(
        params_1.small_lwe_dim,
        params_1.glwe_dim,
        1 << params_1.log_poly_size,
        params_1.base_log_pbs,
        params_1.level_pbs,
        64,
        variance_bsk,
    );
    // norm2 of the small partition
    let (_, _, norm2) = experiments_with_2norms[0];
    let norm2 = 5_f64.sqrt();
    // precision of the big partition
    let (precision, _, _) = experiments_with_2norms[1];
    // some precomputation
    let var_after_dot_product = v_pbs * (norm2 * norm2) as f64;
    let var_modulus_switch = estimate_modulus_switching_noise_with_binary_key(
        params_2.small_lwe_dim,
        params_2.log_poly_size,
        64,
    );

    // optimization of the keyswitch
    let config = CastingConstraint {
        variance_after_dot_product: var_after_dot_product,
        variance_constraint: error::safe_variance_bound_2padbits(precision, 64, p_fail),
        glwe_dim_in: params_1.glwe_dim,
        log_poly_size_in: params_1.log_poly_size,
        security_level: 128,
        variance_modulus_switch: var_modulus_switch,
        small_lwe_dim_out: params_2.small_lwe_dim,
    };

    let ks_search_space = CastingSearchSpace {
        range_base_log_ks: MyRange(1, 53),
        range_level_ks: MyRange(1, 53),
    };

    let interm = config.brute_force(ks_search_space.clone().iter(precision, minimal_ms_value));

    println!("res: {:#?}", res);
    println!("keyswitch: {:#?}", interm);
    let duration = start.elapsed();
    println!(
        "Optimization took: {:?} min",
        duration.as_secs() as f64 / 60.
    );
    // write_to_file(writer, &res).unwrap();
}
