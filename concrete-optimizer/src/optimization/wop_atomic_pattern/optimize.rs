use crate::computing_cost::complexity::Complexity;
use crate::computing_cost::operators::atomic_pattern as complexity_atomic_pattern;
use crate::computing_cost::operators::keyswitch_lwe::KeySwitchLWEComplexity;
use crate::computing_cost::operators::pbs::PbsComplexity;
use crate::noise_estimator::error::{
    error_probability_of_sigma_scale, sigma_scale_of_error_probability,
};

use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::noise_estimator::operators::wop_atomic_pattern::estimate_packing_private_keyswitch;

use crate::optimization::atomic_pattern;
use crate::optimization::wop_atomic_pattern::pareto::{
    BR_BL, BR_BL_FOR_CB, CB_V1_BL, KS_BL, KS_BL_FOR_CB,
};
use crate::parameters::{
    GlweParameters, KeyswitchParameters, KsDecompositionParameters, LweDimension, PbsParameters,
};
use crate::security;
use crate::utils::square;

use complexity_atomic_pattern::DEFAULT as DEFAULT_COMPLEXITY;
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::UnsignedInteger;
use std::collections::HashMap;

pub fn find_p_error(kappa: f64, variance_max: f64, current_maximum_noise: f64) -> f64 {
    let sigma = Variance(variance_max).get_standard_dev() * kappa;
    let sigma_scale = sigma / Variance(current_maximum_noise).get_standard_dev();
    error_probability_of_sigma_scale(sigma_scale)
}

#[derive(Clone, Debug)]
pub struct OptimizationState {
    pub best_solution: Option<Solution>,
    pub count_domain: usize,
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub input_lwe_dimension: u64,
    //n_big
    pub internal_ks_output_lwe_dimension: u64,
    //n_small
    pub ks_decomposition_level_count: u64,
    //l(KS)
    pub ks_decomposition_base_log: u64,
    //b(KS)
    pub glwe_polynomial_size: u64,
    //N
    pub glwe_dimension: u64,
    //k
    pub br_decomposition_level_count: u64,
    //l(BR)
    pub br_decomposition_base_log: u64,
    //b(BR)
    pub complexity: f64,
    pub noise_max: f64,
    pub p_error: f64,
    // error probability
    pub cb_decomposition_level_count: Option<u64>,
    pub cb_decomposition_base_log: Option<u64>,
}

impl Solution {
    pub fn init() -> Self {
        Self {
            input_lwe_dimension: 0,
            internal_ks_output_lwe_dimension: 0,
            ks_decomposition_level_count: 0,
            ks_decomposition_base_log: 0,
            glwe_polynomial_size: 0,
            glwe_dimension: 0,
            br_decomposition_level_count: 0,
            br_decomposition_base_log: 0,
            complexity: 0.,
            noise_max: 0.0,
            p_error: 0.0,
            cb_decomposition_level_count: None,
            cb_decomposition_base_log: None,
        }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct Tab {
    pbs: HashMap<(u64, u64, u64, u64, u64), (f64, Complexity)>,
    modulus_switching: HashMap<(u64, u64), (f64, f64)>,
    key_switching: HashMap<(u64, u64, u64), Vec<(KsDecompositionParameters, (f64, Complexity))>>,
    // NEW VALUE MEMOIZED
    pp_switching: HashMap<(u64, u64, u64, u64), (f64, Complexity)>,
}

#[allow(clippy::too_many_lines)]
pub fn tabulate_circuit_bootstrap<W: UnsignedInteger>(
    security_level: u64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
) -> Tab {
    assert_eq!(security_level, 128);
    assert!(0.0 < maximum_acceptable_error_probability);
    assert!(maximum_acceptable_error_probability < 1.0);

    let ciphertext_modulus_log = W::BITS as u64;
    let mut noise_cost_pbs = HashMap::new();
    let mut noise_cost_modulus_switching = HashMap::new();
    let mut noise_cost_key_switching = HashMap::new();
    let mut noise_cost_pp_switching = HashMap::new();

    for &glwe_dim in glwe_dimensions {
        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
            assert!(8 <= glwe_log_poly_size);
            assert!(glwe_log_poly_size < 18);
            let glwe_poly_size = 1 << glwe_log_poly_size;

            if glwe_dim * glwe_poly_size <= 1 << 13 {
                let glwe_params = GlweParameters {
                    log2_polynomial_size: glwe_log_poly_size,
                    glwe_dimension: glwe_dim,
                };
                let variance_bsk = security::glwe::minimal_variance(
                    glwe_params,
                    ciphertext_modulus_log,
                    security_level,
                );

                for &internal_dim in internal_lwe_dimensions {
                    assert!(256 < internal_dim);

                    let macro_key = (glwe_dim, glwe_poly_size, internal_dim);

                    let variance_ksk = noise_atomic_pattern::variance_ksk(
                        internal_dim,
                        ciphertext_modulus_log,
                        security_level,
                    );
                    let noise_modulus_switching =
                        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
                            internal_dim,
                            glwe_params.polynomial_size(),
                        )
                            .get_variance();
                    let _ = noise_cost_modulus_switching.insert(
                        (internal_dim, glwe_poly_size),
                        (noise_modulus_switching, 0.),
                    );

                    for &br_decomposition_parameter in BR_BL.iter() {
                        let pbs_parameters = PbsParameters {
                            internal_lwe_dimension: LweDimension(internal_dim),
                            br_decomposition_parameter,
                            output_glwe_params: glwe_params,
                        };

                        let complexity_pbs = DEFAULT_COMPLEXITY
                            .pbs
                            .complexity(pbs_parameters, ciphertext_modulus_log);

                        // let complexity_cmux =
                        //     complexity_pbs / (pbs_parameters.internal_lwe_dimension.0 as f64);

                        // PBS of the first layer of the CB
                        let base_noise = noise_atomic_pattern::variance_bootstrap::<W>(
                            pbs_parameters,
                            ciphertext_modulus_log,
                            variance_bsk,
                        )
                        .get_variance();

                        let _ = noise_cost_pbs.insert(
                            (
                                glwe_dim,
                                glwe_poly_size,
                                internal_dim,
                                br_decomposition_parameter.log2_base,
                                br_decomposition_parameter.level,
                            ),
                            (base_noise, complexity_pbs),
                        );
                    }

                    let mut ks_seq = Vec::with_capacity(KS_BL_FOR_CB.len());
                    for &ks_decomposition_parameter in KS_BL_FOR_CB.iter() {
                        let keyswitch_parameter = KeyswitchParameters {
                            input_lwe_dimension: LweDimension(glwe_poly_size * glwe_dim),
                            output_lwe_dimension: LweDimension(internal_dim),
                            ks_decomposition_parameter,
                        };
                        let complexity_keyswitch = DEFAULT_COMPLEXITY
                            .ks_lwe
                            .complexity(keyswitch_parameter, ciphertext_modulus_log);
                        // Keyswitch before bootstrap
                        let noise_keyswitch = noise_atomic_pattern::variance_keyswitch::<W>(
                            keyswitch_parameter,
                            ciphertext_modulus_log,
                            variance_ksk,
                        )
                        .get_variance();
                        ks_seq.push((
                            ks_decomposition_parameter,
                            (noise_keyswitch, complexity_keyswitch),
                        ));
                    }
                    std::mem::drop(noise_cost_key_switching.insert(macro_key, ks_seq));

                    // let pp_ks_decomposition_parameter = pbs_parameters.br_decomposition_parameter;
                    for &pp_ks_decomposition_parameter in BR_BL.iter() {
                        let ppks_parameter = PbsParameters {
                            internal_lwe_dimension: LweDimension(
                                glwe_params.glwe_dimension * glwe_params.polynomial_size(),
                            ),
                            br_decomposition_parameter: pp_ks_decomposition_parameter,
                            output_glwe_params: glwe_params,
                        };
                        // We assume the packing KS and theexternal product in a PBSto have
                        // the same parameters (base, level)
                        let noise_private_packing_ks = estimate_packing_private_keyswitch::<W>(
                            Variance(0.),
                            variance_bsk,
                            ppks_parameter,
                        )
                        .get_variance();

                        let ppks_parameter_complexity = KeyswitchParameters {
                            input_lwe_dimension: LweDimension(
                                glwe_params.glwe_dimension * glwe_params.polynomial_size(),
                            ),
                            output_lwe_dimension: LweDimension(
                                glwe_params.glwe_dimension * glwe_params.polynomial_size(),
                            ),
                            ks_decomposition_parameter: KsDecompositionParameters {
                                level: pp_ks_decomposition_parameter.level,
                                log2_base: pp_ks_decomposition_parameter.log2_base,
                            },
                        };
                        let complexity_ppks = DEFAULT_COMPLEXITY
                            .ks_lwe
                            .complexity(ppks_parameter_complexity, ciphertext_modulus_log);
                        let _ = noise_cost_pp_switching.insert(
                            (
                                glwe_dim,
                                glwe_poly_size,
                                pp_ks_decomposition_parameter.log2_base,
                                pp_ks_decomposition_parameter.level,
                            ),
                            (noise_private_packing_ks, complexity_ppks),
                        );
                    }
                }
            }
        }
    }
    Tab {
        pbs: noise_cost_pbs,
        modulus_switching: noise_cost_modulus_switching,
        key_switching: noise_cost_key_switching,
        pp_switching: noise_cost_pp_switching,
    }
}

const BITS_PADDING_WITHOUT_NOISE: u64 = 1;

#[allow(clippy::expect_fun_call)]
#[allow(clippy::identity_op)]
#[allow(clippy::too_many_lines)]
pub fn optimise_one_with_memo<W: UnsignedInteger>(
    precision: u64, // max precision of a word
    log_norm: f64,  // ?? norm2 of noise multisum, complexity of multisum is neglected
    _security_level: u64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
    n_functions_log: u64, // Many functions at the same time, stay at 1 for start
    memo: &Tab,
    n_inputs: u64, // Tau (nb blocks)
) -> OptimizationState {
    assert!(n_functions_log == 0); // update complexity scaling
    assert!(0.0 < maximum_acceptable_error_probability);
    assert!(maximum_acceptable_error_probability < 1.0);

    let ciphertext_modulus_log = W::BITS as u64;

    // Circuit BS bound
    // 1 bit of message only here =)
    let no_noise_bits = 0 + 1 + BITS_PADDING_WITHOUT_NOISE;
    let noise_bits = ciphertext_modulus_log - no_noise_bits;
    let fatal_noise_limit = (1_u64 << noise_bits) as f64;
    let kappa: f64 = sigma_scale_of_error_probability(maximum_acceptable_error_probability);
    let safe_sigma = fatal_noise_limit / kappa;
    // Bound for first bit extract in BitExtract (dominate others)
    let variance_max = Variance::from_modular_variance::<W>(square(safe_sigma)).get_variance();

    let mut state = OptimizationState {
        best_solution: None,
        count_domain: glwe_dimensions.len()
            * glwe_log_polynomial_sizes.len()
            * internal_lwe_dimensions.len()
            * KS_BL.len()
            * BR_BL.len(),
    };

    let mut best_complexity = f64::INFINITY;

    for &glwe_dim in glwe_dimensions {
        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
            let glwe_poly_size = 1 << glwe_log_poly_size;

            if glwe_dim * glwe_poly_size <= 1 << 13 {
                // Manual experimental CUT
                let glwe_params = GlweParameters {
                    log2_polynomial_size: glwe_log_poly_size,
                    glwe_dimension: glwe_dim,
                };

                for &internal_dim in internal_lwe_dimensions {
                    let &(noise_modulus_switching, _) = memo
                        .modulus_switching
                        .get(&(internal_dim, glwe_poly_size))
                        .expect(&format!(
                            "Internal_dim : {} ; glwe poly size: {}",
                            internal_dim, glwe_poly_size
                        ));

                    let macro_key = (glwe_dim, glwe_poly_size, internal_dim);

                    // BlindRotate dans Circuit BS
                    for &br_decomposition_parameter in BR_BL_FOR_CB.iter() {
                        // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
                        // TODO: choisir indépendemment(separate FP-KS)
                        let pbs_parameters = PbsParameters {
                            internal_lwe_dimension: LweDimension(internal_dim),
                            br_decomposition_parameter,
                            output_glwe_params: glwe_params,
                        };

                        let &(base_noise, complexity_pbs) = memo
                            .pbs
                            .get(&(
                                glwe_dim,
                                glwe_poly_size,
                                internal_dim,
                                br_decomposition_parameter.log2_base,
                                br_decomposition_parameter.level,
                            ))
                            .unwrap();

                        // new pbs key for the bit extract pbs, shared
                        let bit_extract_decomposition_parameter = br_decomposition_parameter;

                        let &(_bit_extract_base_noise, complexity_bit_extract_pbs) = memo
                            .pbs
                            .get(&(
                                glwe_dim,
                                glwe_poly_size,
                                internal_dim,
                                bit_extract_decomposition_parameter.log2_base,
                                bit_extract_decomposition_parameter.level,
                            ))
                            .unwrap();

                        // private packing keyswitch, <=> FP-KS (Circuit Boostrap)
                        let pp_ks_decomposition_parameter =
                            pbs_parameters.br_decomposition_parameter;
                        // for &pp_ks_decomposition_parameter in PP_KS_BL.iter() { // independant params for FP-KS

                        // Circuit Boostrap
                        let &(base_noise_private_packing_ks, complexity_ppks) = memo
                            .pp_switching
                            .get(&(
                                glwe_dim,
                                glwe_poly_size,
                                pp_ks_decomposition_parameter.log2_base,
                                pp_ks_decomposition_parameter.level,
                            ))
                            .expect(&format!(
                                "{}, {}, {}, {}",
                                glwe_dim,
                                glwe_poly_size,
                                pp_ks_decomposition_parameter.log2_base,
                                pp_ks_decomposition_parameter.level,
                            ));

                        // CircuitBootstrap: new parameters l,b
                        for &circuit_pbs_decomposition_parameter in CB_V1_BL.iter() {
                            // Hybrid packing
                            let nb_cmux = 1_u64;
                            let cmux_tree_blind_rotate_parameters = PbsParameters {
                                internal_lwe_dimension: LweDimension(nb_cmux), // complexity for 1 cmux
                                br_decomposition_parameter: circuit_pbs_decomposition_parameter,
                                output_glwe_params: pbs_parameters.output_glwe_params,
                            };

                            // Hybrid packing (rename)
                            let complexity_cmux_for_cb = DEFAULT_COMPLEXITY.pbs.complexity(
                                cmux_tree_blind_rotate_parameters,
                                ciphertext_modulus_log,
                            ); // TODO: missing fft transform

                            // Hybrid packing (Do we have 1 or 2 groups)
                            #[allow(clippy::precedence)]
                            let complexity_cmux_tree = if precision * n_inputs as u64 // sum of precisions
                                > pbs_parameters.output_glwe_params.log2_polynomial_size
                            {
                                // 2 groups
                                complexity_cmux_for_cb
                                    * (1 << (precision * n_inputs
                                        - pbs_parameters.output_glwe_params.log2_polynomial_size)
                                        - 1) as f64
                                // * (f64::exp2(
                                //     (precision * n_inputs
                                //         - pbs_parameters
                                //             .output_glwe_params
                                //             .log2_polynomial_size)
                                //         as f64,
                                // ) - 1.)
                            } else {
                                // 1 group, no cmux tree
                                0.
                            };
                            // Hybrid packing blind rotate
                            let complexity_g_br = complexity_cmux_for_cb
                                * f64::min(
                                    (pbs_parameters.output_glwe_params.log2_polynomial_size) as f64,
                                    (precision * n_inputs) as f64,
                                );

                            let noise_private_packing_ks =
                                base_noise_private_packing_ks + base_noise / 2.;

                            // Circuit Boostrap
                            if noise_private_packing_ks + noise_modulus_switching > variance_max
                                || (precision - 1) as f64 * complexity_pbs + complexity_ppks
                                    > best_complexity
                            {
                                continue;
                            }

                            let noise_ggsw = noise_private_packing_ks;

                            let noise_one_external_product_for_cmux_tree =
                                noise_atomic_pattern::variance_bootstrap::<W>(
                                    cmux_tree_blind_rotate_parameters,
                                    ciphertext_modulus_log,
                                    Variance::from_variance(noise_ggsw),
                                )
                                .get_variance();

                            // all fp-ks
                            let complexity_all_ppks =
                                ((pbs_parameters.output_glwe_params.glwe_dimension + 1)
                                    * circuit_pbs_decomposition_parameter.level
                                    * precision) as f64
                                    * complexity_ppks;

                            // final out noise hybrid packing
                            let noise_cmux_tree_blind_rotate =
                                noise_one_external_product_for_cmux_tree
                                    * (precision * n_inputs) as f64;

                            let noise_multisum =
                                (2_f64.powf(2. * log_norm as f64)) * noise_cmux_tree_blind_rotate; // out noise * weights

                            // Shared by all pbs (like brs)
                            let key_switching_q = memo.key_switching.get(&macro_key).unwrap();
                            for &(
                                ks_decomposition_parameter,
                                (noise_keyswitch, complexity_keyswitch),
                            ) in key_switching_q.iter()
                            {
                                let keyswitch_parameter = KeyswitchParameters {
                                    input_lwe_dimension: LweDimension(glwe_poly_size * glwe_dim),
                                    output_lwe_dimension: LweDimension(internal_dim),
                                    ks_decomposition_parameter,
                                };
                                let complexity_all_ks = precision as f64 * complexity_keyswitch;
                                if noise_private_packing_ks
                                    + noise_modulus_switching
                                    + noise_keyswitch
                                    > variance_max
                                    || (precision - 1) as f64 * complexity_pbs
                                        + complexity_ppks
                                        + precision as f64 * complexity_keyswitch
                                        > best_complexity
                                {
                                    continue;
                                }

                                // noise_multisum = dot
                                let current_maximal_noise = noise_multisum
                                    * (1 << (2 * (precision - 1))) as f64
                                    + noise_keyswitch
                                    + noise_modulus_switching;

                                let complexity_all_pbs =
                                    (precision * circuit_pbs_decomposition_parameter.level) as f64
                                        * complexity_pbs
                                        + (precision - 1) as f64 * complexity_bit_extract_pbs;

                                let complexity_bias =
                                    (complexity_all_ppks + complexity_all_pbs + complexity_all_ks)
                                        * n_inputs as f64;

                                let complexity_slope = complexity_cmux_tree + complexity_g_br;

                                let current_complexity = complexity_slope
                                    * (1 << n_functions_log) as f64
                                    + complexity_bias;

                                if current_complexity > best_complexity {
                                    // next level is more costly
                                    break;
                                }

                                if current_complexity < best_complexity
                                    && current_maximal_noise < variance_max
                                {
                                    best_complexity = current_complexity;
                                    state.best_solution = Some(Solution {
                                        input_lwe_dimension: pbs_parameters
                                            .output_glwe_params
                                            .glwe_dimension
                                            * pbs_parameters.output_glwe_params.polynomial_size(),
                                        internal_ks_output_lwe_dimension: keyswitch_parameter
                                            .output_lwe_dimension
                                            .0,
                                        ks_decomposition_level_count: keyswitch_parameter
                                            .ks_decomposition_parameter
                                            .level,
                                        ks_decomposition_base_log: keyswitch_parameter
                                            .ks_decomposition_parameter
                                            .log2_base,
                                        glwe_polynomial_size: pbs_parameters
                                            .output_glwe_params
                                            .polynomial_size(),
                                        glwe_dimension: pbs_parameters
                                            .output_glwe_params
                                            .glwe_dimension,
                                        br_decomposition_level_count: pbs_parameters
                                            .br_decomposition_parameter
                                            .level,
                                        br_decomposition_base_log: pbs_parameters
                                            .br_decomposition_parameter
                                            .log2_base,
                                        noise_max: current_maximal_noise,
                                        complexity: current_complexity,
                                        p_error: find_p_error(
                                            kappa,
                                            variance_max,
                                            current_maximal_noise,
                                        ), // consts.maximum_acceptable_error_probability,
                                        cb_decomposition_level_count: Some(
                                            circuit_pbs_decomposition_parameter.level,
                                        ),
                                        cb_decomposition_base_log: Some(
                                            circuit_pbs_decomposition_parameter.log2_base,
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    state
}

#[allow(clippy::too_many_lines)]
pub fn optimize_one<W: UnsignedInteger>(
    _sum_size: u64,
    precision: u64,
    security_level: u64,
    noise_factor: f64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
    memo_opt: &mut Option<Tab>,
) -> atomic_pattern::OptimizationState {
    // Basic heuristic to split in several word
    let no_sol = atomic_pattern::OptimizationState {
        best_solution: None,
        count_domain: 0,
    };
    #[allow(clippy::match_same_arms)]
    let (nb_words, max_word_precision) = match precision {
        1 => (1, 1),
        2 => (1, 2),
        3 => (2, 2),
        4 => (2, 3),
        5 => (2, 3),
        6 => (3, 3),
        7 => (3, 3),
        8 => (3, 3),
        9 => (3, 4),
        10 => (3, 4),
        11 => (3, 4),
        12 => (4, 4),
        13 => (4, 4),
        14 => (4, 4),
        15 => (4, 4),
        16 => (4, 5),
        _ => return no_sol,
    };
    let log_norm = noise_factor.log2();
    let n_functions_log = 0;
    let memo = memo_opt.get_or_insert_with(|| {
        tabulate_circuit_bootstrap::<W>(
            security_level,
            maximum_acceptable_error_probability,
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        )
    });
    let result = optimise_one_with_memo::<W>(
        max_word_precision,
        log_norm,
        security_level,
        maximum_acceptable_error_probability,
        glwe_log_polynomial_sizes,
        glwe_dimensions,
        internal_lwe_dimensions,
        n_functions_log,
        memo,
        nb_words, // Tau
    );
    let best_solution = result.best_solution.map(|sol| atomic_pattern::Solution {
        input_lwe_dimension: sol.input_lwe_dimension,
        internal_ks_output_lwe_dimension: sol.internal_ks_output_lwe_dimension,
        ks_decomposition_level_count: sol.ks_decomposition_level_count,
        ks_decomposition_base_log: sol.ks_decomposition_base_log,
        glwe_polynomial_size: sol.glwe_polynomial_size,
        glwe_dimension: sol.glwe_dimension,
        br_decomposition_level_count: sol.br_decomposition_level_count,
        br_decomposition_base_log: sol.br_decomposition_base_log,
        complexity: sol.complexity,
        lut_complexity: sol.complexity,
        noise_max: sol.noise_max,
        p_error: sol.p_error,
    });
    atomic_pattern::OptimizationState {
        best_solution,
        count_domain: result.count_domain,
    }
}
