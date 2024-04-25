#![allow(clippy::too_many_arguments)]

use std::cmp::Ordering;

use crate::implementation::{from_torus, zip_eq};
use concrete_cpu_noise_model::gaussian_noise::noise::blind_rotate::variance_blind_rotate;
use concrete_cpu_noise_model::gaussian_noise::noise::keyswitch::variance_keyswitch;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;
use concrete_cpu_noise_model::gaussian_noise::noise::private_packing_keyswitch::estimate_packing_private_keyswitch;
use concrete_csprng::generators::SoftwareRandomGenerator;
use concrete_security_curves::gaussian::security::{minimal_variance_glwe, minimal_variance_lwe};
use tfhe::core_crypto::commons::math::random::RandomGenerator;
use tfhe::core_crypto::commons::parameters::*;

use tfhe::core_crypto::entities::{Polynomial, PolynomialList};

pub fn random_gaussian_pair(
    variance: f64,
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) -> (f64, f64) {
    let mut buff = [0_f64, 0_f64];
    csprng.fill_slice_with_random_gaussian(buff.as_mut_slice(), 0.0, variance);
    (buff[0], buff[1])
}

pub fn modular_add(ct_1: u64, ct_2: u64, modulus: u64) -> u64 {
    let mut res = ct_1.wrapping_add(ct_2);
    if res >= modulus {
        res -= modulus;
    }
    res
}

fn integer_round(lwe: u64, log_poly_size: u64, ciphertext_modulus_log: usize) -> u64 {
    let input = lwe;
    let non_rep_bit_count: usize = ciphertext_modulus_log - log_poly_size as usize - 1;
    // We generate a mask which captures the non representable bits
    let non_rep_mask = 1_u64 << (non_rep_bit_count - 1);
    // We retrieve the non representable bits
    let non_rep_bits = input & non_rep_mask;
    // We extract the msb of the  non representable bits to perform the rounding
    let non_rep_msb = non_rep_bits >> (non_rep_bit_count - 1);
    // We remove the non-representable bits and perform the rounding
    let res = input >> non_rep_bit_count;
    ((res + non_rep_msb) << non_rep_bit_count) >> non_rep_bit_count
}

fn bool_round(lwe: u64, ciphertext_modulus_log: usize) -> u64 {
    integer_round(lwe, 0, ciphertext_modulus_log)
}

/// Function to extract `number_of_bits_to_extract` from an [`LweCiphertext`] starting at the bit
/// number `delta_log` (0-indexed) included.
///
/// Output bits are ordered from the MSB to the LSB. Each one of them is output in a distinct LWE
/// ciphertext, containing the encryption of the bit scaled by q/2 (i.e., the most significant bit
/// in the plaintext representation).
pub fn extract_bits(
    lwe_list_out: &mut [u64],
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
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) {
    let polynomial_size = 1 << log_poly_size;
    let mut lookup_table = vec![0_u64; polynomial_size as usize];
    let ciphertext_n_bits = u64::BITS as usize;

    debug_assert!(
        ciphertext_n_bits >= number_of_bits_to_extract + delta_log,
        "Tried to extract {} bits, while the maximum number of extractable bits for {} bits
        ciphertexts and a scaling factor of 2^{} is {}",
        number_of_bits_to_extract,
        ciphertext_n_bits,
        delta_log,
        ciphertext_n_bits - delta_log,
    );

    let mut temp_lwe = lwe_in;
    // We iterate on the list in reverse as we want to store the extracted MSB at index 0
    for (bit_idx, output_ct) in lwe_list_out.iter_mut().rev().enumerate() {
        let shifted_lwe = temp_lwe << (ciphertext_n_bits - delta_log - bit_idx - 1);
        let variance_ksk =
            minimal_variance_lwe(lwe_dimension, ciphertext_modulus_log, security_level);
        let keyswitch_variance = variance_keyswitch(
            glwe_dimension * polynomial_size,
            ks_log_base,
            ks_level,
            ciphertext_modulus_log,
            variance_ksk,
        );
        let (keyswitch_noise, _) = random_gaussian_pair(keyswitch_variance, csprng);

        // Key switch to input PBS key
        let keyswitched_shifted_lwe = shifted_lwe.wrapping_add(from_torus(keyswitch_noise));

        // Store the keyswitch output unmodified to the output list (as we need to to do other
        // computations on the output of the keyswitch)
        *output_ct = keyswitched_shifted_lwe;

        // If this was the last extracted bit, break
        // we subtract 1 because if the number_of_bits_to_extract is 1 we want to stop right away
        if bit_idx == number_of_bits_to_extract - 1 {
            break;
        }

        // Add q/4 to center the error while computing a negacyclic LUT
        let corrected_lwe = keyswitched_shifted_lwe.wrapping_add(1_u64 << (ciphertext_n_bits - 2));

        // Fill lut for the current bit (equivalent to trivial encryption as mask is 0s)
        // The LUT is filled with -alpha in each coefficient where alpha = delta*2^{bit_idx-1}
        for poly_coeff in lookup_table.iter_mut() {
            *poly_coeff = (1_u64 << (delta_log - 1 + bit_idx)).wrapping_neg();
        }

        // modulus switch
        let modulus_switch_variance = estimate_modulus_switching_noise_with_binary_key(
            lwe_dimension,
            log_poly_size,
            ciphertext_modulus_log,
        );
        let (modulus_switch_noise, _) = random_gaussian_pair(modulus_switch_variance, csprng);

        let modulus_switched_lwe = modular_add(
            integer_round(
                corrected_lwe,
                log_poly_size,
                ciphertext_modulus_log as usize,
            ),
            integer_round(
                from_torus(modulus_switch_noise),
                log_poly_size,
                ciphertext_modulus_log as usize,
            ),
            2 * polynomial_size,
        );

        // blind rotate + sample extract
        let variance_bsk = minimal_variance_glwe(
            glwe_dimension,
            polynomial_size,
            ciphertext_modulus_log,
            security_level,
        );
        let blind_rotate_variance = variance_blind_rotate(
            lwe_dimension,
            glwe_dimension,
            polynomial_size,
            br_log_base,
            br_level,
            ciphertext_modulus_log,
            53,
            variance_bsk,
        );
        let (blind_rotate_noise, _) = random_gaussian_pair(blind_rotate_variance, csprng);

        let blind_rotated_lwe = if modulus_switched_lwe < polynomial_size {
            lookup_table[modulus_switched_lwe as usize].wrapping_add(from_torus(blind_rotate_noise))
        } else {
            lookup_table[(modulus_switched_lwe - polynomial_size) as usize]
                .wrapping_neg()
                .wrapping_add(from_torus(blind_rotate_noise))
        };

        // Add alpha where alpha = delta*2^{bit_idx-1} to end up with an encryption of 0 if the
        // extracted bit was 0 and 1 in the other case
        let corrected_blind_rotated_lwe =
            blind_rotated_lwe.wrapping_add(1_u64 << (delta_log + bit_idx - 1));

        // Remove the extracted bit from the initial LWE to get a 0 at the extracted bit location.
        temp_lwe = temp_lwe.wrapping_sub(corrected_blind_rotated_lwe);
    }
}

/// Circuit bootstrapping for boolean messages, i.e. containing only one bit of message
///
/// The output GGSW ciphertext `ggsw_out` decomposition base log and level count are used as the
/// circuit_bootstrap_boolean decomposition base log and level count.
pub fn circuit_bootstrap_boolean(
    lwe_in: u64,
    delta_log: usize,
    log_poly_size: u64,
    lwe_dimension: u64,
    ciphertext_modulus_log: u32,
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) -> u64 {
    //  homomorphic_shift_boolean outputs the LUT evaluation without the blind rotate noise
    // nor the packing keyswitch noise. There will be added latter during the vertical packing.
    homomorphic_shift_boolean(
        lwe_in,
        lwe_dimension,
        log_poly_size,
        delta_log,
        ciphertext_modulus_log,
        csprng,
    )
}

/// Homomorphic shift for LWE without padding bit
///
/// Starts by shifting the message bit at bit #delta_log to the padding bit and then shifts it to
/// the right by base_log * level.
pub fn homomorphic_shift_boolean(
    lwe_in: u64,
    lwe_dimension: u64,
    log_poly_size: u64,
    delta_log: usize,
    ciphertext_modulus_log: u32,
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) -> u64 {
    let ciphertext_n_bits = ciphertext_modulus_log;
    let polynomial_size = 1 << log_poly_size;

    // Shift message LSB on padding bit, at this point we expect to have messages with only 1 bit
    // of information
    let shift = 1 << (ciphertext_n_bits - delta_log as u32 - 1);
    debug_assert_eq!(shift, 1);

    let shifted_lwe = lwe_in.wrapping_mul(shift);

    // Add q/4 to center the error while computing a negacyclic LUT
    let corrected_lwe = shifted_lwe.wrapping_add(1_u64 << (ciphertext_n_bits - 2));
    let mut lookup_table = vec![0_u64; polynomial_size as usize];

    // Fill lut
    // The LUT is filled with -alpha in each coefficient where
    // alpha = 2^{log(q) - 1 - 1}
    // in reality, the algorithm uses alpha = 2^{log(q) - 1 - base_log * level} but here
    // we only care about simulating the computation
    let alpha = 1_u64 << (ciphertext_n_bits - 1 - 1);

    for body in lookup_table.iter_mut() {
        *body = alpha.wrapping_neg();
    }

    // Applying a negacyclic LUT on a ciphertext with one bit of message in the MSB and no bit
    // of padding

    // modulus switch
    let modulus_switch_variance = estimate_modulus_switching_noise_with_binary_key(
        lwe_dimension,
        log_poly_size,
        ciphertext_modulus_log,
    );
    let (modulus_switch_noise, _) = random_gaussian_pair(modulus_switch_variance, csprng);

    let modulus_switched_lwe = modular_add(
        integer_round(
            corrected_lwe,
            log_poly_size,
            ciphertext_modulus_log as usize,
        ),
        integer_round(
            from_torus(modulus_switch_noise),
            log_poly_size,
            ciphertext_modulus_log as usize,
        ),
        2 * polynomial_size,
    );

    // noiseless blind rotate + sample extract
    let blind_rotated_lwe = if modulus_switched_lwe < polynomial_size {
        lookup_table[modulus_switched_lwe as usize]
    } else {
        lookup_table[(modulus_switched_lwe - polynomial_size) as usize].wrapping_neg()
    };

    // Add alpha where alpha = 2^{log(q) - 1 - base_log * level}
    // To end up with an encryption of 0 if the message bit was 0 and 1 in the other case
    blind_rotated_lwe.wrapping_add(1_u64 << (ciphertext_n_bits - 1 - 1))
}

pub fn cmux_tree_memory_optimized(
    log_poly_size: u64,
    output_glwe: &mut [u64],
    lut_per_layer: PolynomialList<&[u64]>,
    ggsw_list: &[u64],
    ciphertext_modulus_log: u32,
) {
    let polynomial_size = 1 << log_poly_size;
    let nb_layer = ggsw_list.len();

    // These are accumulator that will be used to propagate the result from layer to layer
    // At index 0 you have the lut that will be loaded, and then the result for each layer gets
    // computed at the next index, last layer result gets stored in `result`.
    // This allow to use memory space in C * nb_layer instead of C' * 2 ^ nb_layer
    let mut t_0_data = vec![vec![0_u64; polynomial_size as usize]; nb_layer];
    let mut t_1_data = vec![vec![0_u64; polynomial_size as usize]; nb_layer];

    let mut t_fill = vec![0_u64; nb_layer];

    let polynomial_size = lut_per_layer.polynomial_size();
    let mut lut_polynomial_iter = lut_per_layer
        .into_container()
        .chunks_exact(polynomial_size.0)
        .map(Polynomial::from_container);
    loop {
        let even = lut_polynomial_iter.next();
        let odd = lut_polynomial_iter.next();

        let (lut_2i, lut_2i_plus_1) = match (even, odd) {
            (Some(even), Some(odd)) => (even, odd),
            _ => break,
        };

        let mut t_iter = zip_eq(t_0_data.iter_mut(), t_1_data.iter_mut()).enumerate();

        let (mut j_counter, (mut t0_j, mut t1_j)) = t_iter.next().unwrap();

        t0_j.copy_from_slice(lut_2i.into_container());

        t1_j.copy_from_slice(lut_2i_plus_1.into_container());

        t_fill[0] = 2;

        for (j, ggsw) in ggsw_list.iter().rev().enumerate() {
            if t_fill[j] == 2 {
                let diff_lut: Vec<u64> = t1_j
                    .iter()
                    .zip(t0_j.iter())
                    .map(|(a, b)| a.wrapping_sub(*b))
                    .collect();

                if j != nb_layer - 1 {
                    let (j_counter_plus_1, (t_0_j_plus_1, t_1_j_plus_1)) = t_iter.next().unwrap();

                    debug_assert_eq!(j_counter, j);
                    debug_assert_eq!(j_counter_plus_1, j + 1);

                    let output: &mut Vec<u64> = if t_fill[j + 1] == 0 {
                        t_0_j_plus_1
                    } else {
                        t_1_j_plus_1
                    };

                    // Computing (t1_j - t0_j) * bit_in_ggsw + t0_j which will output t1_j if
                    // bit_in_ggsw == 1 and t0_j if bit_in_ggsw == 0
                    output.copy_from_slice(t0_j);
                    let bit_in_ggsw = bool_round(*ggsw, ciphertext_modulus_log as usize);
                    assert!(bit_in_ggsw == 0 || bit_in_ggsw == 1);
                    for (o, diff) in output.iter_mut().zip(diff_lut.iter()) {
                        *o = o.wrapping_add(diff.wrapping_mul(bit_in_ggsw));
                    }
                    t_fill[j + 1] += 1;
                    t_fill[j] = 0;

                    (j_counter, t0_j, t1_j) = (j_counter_plus_1, t_0_j_plus_1, t_1_j_plus_1);
                } else {
                    // Computing (t1_j - t0_j) * bit_in_ggsw + t0_j which will output t1_j if
                    // bit_in_ggsw == 1 and t0_j if bit_in_ggsw == 0
                    output_glwe.copy_from_slice(t0_j);
                    let bit_in_ggsw = bool_round(*ggsw, ciphertext_modulus_log as usize);
                    assert!(bit_in_ggsw == 0 || bit_in_ggsw == 1);
                    for (o, diff) in output_glwe.iter_mut().zip(diff_lut.iter()) {
                        *o = o.wrapping_add(diff.wrapping_mul(bit_in_ggsw));
                    }
                }
            } else {
                break;
            }
        }
    }
}

fn print_ct(ct: u64) {
    print!("{}", (((ct >> 53) + 1) >> 1) % (1 << 10));
}

fn log2(a: usize) -> usize {
    let result = u64::BITS as usize - 1 - a.leading_zeros() as usize;

    debug_assert_eq!(a, 1 << result);

    result
}

pub fn blind_rotate(
    log_poly_size: u64,
    glwe_dimension: u64,
    lwe_dimension: u64,
    fpks_log_base: u64,
    fpks_level: u64,
    cb_log_base: u64,
    cb_level: u64,
    pbs_log_base: u64,
    pbs_level: u64,
    lookup_table: &mut [u64],
    ggsw_list: &[u64],
    ciphertext_modulus_log: u32,
    security_level: u64,
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) -> u64 {
    let polynomial_size = 1 << log_poly_size;
    let mut monomial_degree = 1;
    let mut exponent = 0_u64;
    // Here we compute exponent = \sum_{i} 2^i bit_in_ggsw_i
    for ggsw in ggsw_list.iter().rev() {
        // retrieve bit
        let bit_in_ggsw = bool_round(*ggsw, ciphertext_modulus_log as usize);
        exponent = exponent.wrapping_add(bit_in_ggsw.wrapping_mul(monomial_degree));
        monomial_degree <<= 1;
    }

    // Vertical Packing + Blind rotate
    // The Vertical packing (VP) selected a lut `lookup_table` using part of the the GGSWs
    // The blind rotate outputs: L[exponent] + noise_blindrotate(n= number of GGSWs, var_bsk=var_circuit_bootstrap)

    let variance_bsk = minimal_variance_glwe(
        glwe_dimension,
        polynomial_size,
        ciphertext_modulus_log,
        security_level,
    );

    let blind_rotate_variance = variance_blind_rotate(
        lwe_dimension,
        glwe_dimension,
        polynomial_size,
        pbs_log_base,
        pbs_level,
        ciphertext_modulus_log,
        53,
        variance_bsk,
    );

    let ppks_variance = estimate_packing_private_keyswitch(
        0.,
        variance_bsk,
        fpks_log_base,
        fpks_level,
        glwe_dimension,
        polynomial_size,
        ciphertext_modulus_log,
    );

    let ggsw_variance = blind_rotate_variance + ppks_variance;

    let vertical_packing_variance = variance_blind_rotate(
        ggsw_list.len() as u64,
        glwe_dimension,
        polynomial_size,
        cb_log_base,
        cb_level,
        ciphertext_modulus_log,
        53,
        ggsw_variance,
    );

    let (vertical_packing_variance, _) = random_gaussian_pair(vertical_packing_variance, csprng);

    if exponent < polynomial_size {
        lookup_table[exponent as usize].wrapping_add(from_torus(vertical_packing_variance))
    } else {
        lookup_table[(exponent - polynomial_size) as usize]
            .wrapping_neg()
            .wrapping_add(from_torus(vertical_packing_variance))
    }
}

// GGSW ciphertexts are stored from the msb (vec_ggsw[0]) to the lsb (vec_ggsw[last])
pub fn vertical_packing(
    lut: &[u64],
    ggsw_list: &[u64],
    log_poly_size: u64,
    glwe_dimension: u64,
    lwe_dimension: u64,
    fpks_log_base: u64,
    fpks_level: u64,
    cb_log_base: u64,
    cb_level: u64,
    pbs_log_base: u64,
    pbs_level: u64,
    ciphertext_modulus_log: u32,
    security_level: u64,
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) -> u64 {
    let polynomial_size = 1 << log_poly_size;

    let log_lut_number = log2(lut.len());
    debug_assert_eq!(
        ggsw_list.len(),
        log_lut_number,
        "ggsw_lst lzn = {} \nlog_lu_nb = {}",
        ggsw_list.len(),
        log_lut_number
    );

    let mut cmux_tree_lut_res = vec![0_u64; polynomial_size];

    let br_ggsw = match log_lut_number.cmp(&(log_poly_size as usize)) {
        Ordering::Less => {
            // No CMUX tree, directly go to blind rotate after having copy the lut
            cmux_tree_lut_res[0..lut.len()].copy_from_slice(lut);
            ggsw_list
        }
        Ordering::Equal => {
            // No CMUX tree, directly go to blind rotate after having copy the lut
            cmux_tree_lut_res.copy_from_slice(lut);
            ggsw_list
        }
        Ordering::Greater => {
            // Need a CMUX tree that will select a smaller lut

            let log_number_of_luts_for_cmux_tree = log_lut_number - log_poly_size as usize;

            // split the vec of GGSW in two, the msb GGSW is for the CMux tree and the lsb GGSW is
            // for the last blind rotation.
            let (cmux_ggsw, br_ggsw) = ggsw_list.split_at(log_number_of_luts_for_cmux_tree);
            debug_assert_eq!(br_ggsw.len(), log_poly_size as usize);

            // extract the smaller luts from the big lut
            let small_luts = PolynomialList::from_container(lut, PolynomialSize(polynomial_size));
            // cmux_tree_memory_optimized will write the right lut in `cmux_tree_lut_res`
            cmux_tree_memory_optimized(
                log_poly_size,
                &mut cmux_tree_lut_res,
                small_luts,
                cmux_ggsw,
                ciphertext_modulus_log,
            );

            br_ggsw
        }
    };

    // take the lut pre-selected with the vertical packing and perform the blind_rotate.
    // the simulation noise of the circuit bootstrap + vertical packing is added in this function
    blind_rotate(
        log_poly_size,
        glwe_dimension,
        lwe_dimension,
        fpks_log_base,
        fpks_level,
        cb_log_base,
        cb_level,
        pbs_log_base,
        pbs_level,
        &mut cmux_tree_lut_res,
        br_ggsw,
        ciphertext_modulus_log,
        security_level,
        csprng,
    )
}

/// Perform a circuit bootstrap followed by a vertical packing on ciphertexts encrypting boolean
/// messages.
///
/// The circuit bootstrapping uses the private functional packing key switch.
///
/// This is supposed to be used only with boolean (1 bit of message) LWE ciphertexts.
///  k,  N,    n, br_l,br_b, ks_l,ks_b, cb_l,cb_b, pp_l,pp_b
pub fn circuit_bootstrap_boolean_vertical_packing(
    lwe_list_in: &[u64],
    lwe_list_out: &mut [u64],
    luts: &[u64],
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
    csprng: &mut RandomGenerator<SoftwareRandomGenerator>,
) {
    let mut ggsw_list = vec![0_u64; lwe_list_in.len()];
    let delta_log = u64::BITS as usize - 1;
    for (lwe_in, ggsw) in zip_eq(lwe_list_in.iter(), ggsw_list.iter_mut()) {
        *ggsw = circuit_bootstrap_boolean(
            *lwe_in,
            delta_log,
            log_poly_size,
            lwe_dimension,
            ciphertext_modulus_log,
            csprng,
        );
    }

    for (lut, lwe_out) in zip_eq(luts.chunks(1 << lwe_list_in.len()), lwe_list_out.iter_mut()) {
        *lwe_out = vertical_packing(
            lut,
            &ggsw_list,
            log_poly_size,
            glwe_dimension,
            lwe_dimension,
            pp_log_base,
            pp_level,
            cb_log_base,
            cb_level,
            pbs_log_base,
            pbs_level,
            ciphertext_modulus_log,
            security_level,
            csprng,
        );
    }
}
