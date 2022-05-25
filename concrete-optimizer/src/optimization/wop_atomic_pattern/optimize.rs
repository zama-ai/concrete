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
use crate::optimization::atomic_pattern::OptimizationDecompositionsConsts;
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

#[derive(Clone, Copy, Debug)]
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
struct NoiseCostByMicroParam {
    pbs: Vec<(f64, Complexity)>,
    key_switching: Vec<(f64, Complexity)>,
    pp_switching: Vec<(f64, Complexity)>,
}

#[allow(clippy::too_many_lines)]
fn compute_noise_cost_by_micro_param<W: UnsignedInteger>(
    security_level: u64,
    glwe_params: GlweParameters,

    internal_dim: u64,
) -> NoiseCostByMicroParam {
    assert!(256 < internal_dim);

    let mut noise_cost_pbs = Vec::new();
    let mut noise_cost_key_switching = Vec::new();
    let mut noise_cost_pp_switching = Vec::new();

    let ciphertext_modulus_log = W::BITS as u64;

    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);

    let variance_ksk =
        noise_atomic_pattern::variance_ksk(internal_dim, ciphertext_modulus_log, security_level);

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

        noise_cost_pbs.push((base_noise, complexity_pbs));
    }

    for &ks_decomposition_parameter in KS_BL_FOR_CB.iter() {
        let keyswitch_parameter = KeyswitchParameters {
            input_lwe_dimension: LweDimension(glwe_params.sample_extract_lwe_dimension()),
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
        noise_cost_key_switching.push((noise_keyswitch, complexity_keyswitch));
    }

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
        let noise_private_packing_ks =
            estimate_packing_private_keyswitch::<W>(Variance(0.), variance_bsk, ppks_parameter)
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
        noise_cost_pp_switching.push((noise_private_packing_ks, complexity_ppks));
    }

    NoiseCostByMicroParam {
        pbs: noise_cost_pbs,
        key_switching: noise_cost_key_switching,
        pp_switching: noise_cost_pp_switching,
    }
}

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions<W: UnsignedInteger>(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    glwe_params: GlweParameters,
    internal_dim: u64,
    n_functions: u64,
    precision: u64,
    n_inputs: u64, // Tau
) {
    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let global_precision = n_inputs * precision;
    let variance_max = consts.safe_variance;
    let log_norm = consts.noise_factor.log2();

    let micro_tab =
        compute_noise_cost_by_micro_param::<W>(consts.security_level, glwe_params, internal_dim);

    let noise_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            internal_dim,
            glwe_params.polynomial_size(),
        )
        .get_variance();

    if noise_modulus_switching > consts.safe_variance {
        return;
    }

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);

    // BlindRotate dans Circuit BS
    for (br_dp_index, &br_decomposition_parameter) in BR_BL_FOR_CB.iter().enumerate() {
        // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
        // TODO: choisir indépendemment(separate FP-KS)
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        let (base_noise, complexity_pbs) = micro_tab.pbs[br_dp_index];

        // new pbs key for the bit extract pbs, shared
        let bit_extract_dp_index = br_dp_index;

        let (_bit_extract_base_noise, complexity_bit_extract_pbs) =
            micro_tab.pbs[bit_extract_dp_index];

        let complexity_bit_extract_wo_ks =
            (n_inputs * (precision - 1)) as f64 * complexity_bit_extract_pbs;

        if complexity_bit_extract_wo_ks > best_complexity {
            continue;
        }

        // private packing keyswitch, <=> FP-KS (Circuit Boostrap)
        let pp_ks_dp_index = br_dp_index;

        // Circuit Boostrap
        let (base_noise_private_packing_ks, complexity_ppks) =
            micro_tab.pp_switching[pp_ks_dp_index];

        // CircuitBootstrap: new parameters l,b
        for &circuit_pbs_decomposition_parameter in CB_V1_BL.iter() {
            // Hybrid packing
            let nb_cmux = 1_u64;
            let cmux_tree_blind_rotate_parameters = PbsParameters {
                internal_lwe_dimension: LweDimension(nb_cmux), // complexity for 1 cmux
                br_decomposition_parameter: circuit_pbs_decomposition_parameter,
                output_glwe_params: pbs_parameters.output_glwe_params,
            };

            // Hybrid packing
            let complexity_1_cmux_hp = DEFAULT_COMPLEXITY
                .pbs
                .complexity(cmux_tree_blind_rotate_parameters, ciphertext_modulus_log); // TODO: missing fft transform

            // Hybrid packing (Do we have 1 or 2 groups)
            let log2_polynomial_size = pbs_parameters.output_glwe_params.log2_polynomial_size;
            // Size of cmux_group, can be zero
            let cmux_group_count = if global_precision > log2_polynomial_size {
                2f64.powi((global_precision - log2_polynomial_size - 1) as i32)
            } else {
                0.0
            };
            let complexity_cmux_tree = cmux_group_count as f64 * complexity_1_cmux_hp;
            // Hybrid packing blind rotate
            let complexity_g_br = complexity_1_cmux_hp
                * u64::min(
                    pbs_parameters.output_glwe_params.log2_polynomial_size,
                    global_precision,
                ) as f64;

            let complexity_hybrid_packing = complexity_cmux_tree + complexity_g_br;
            let complexity_multi_hybrid_packing = n_functions as f64 * complexity_hybrid_packing;

            // Circuit bs: fp-ks
            let complexity_all_ppks = ((pbs_parameters.output_glwe_params.glwe_dimension + 1)
                * circuit_pbs_decomposition_parameter.level
                * precision
                * n_inputs) as f64
                * complexity_ppks;

            // Circuit bs: pbs
            let complexity_all_pbs =
                (n_inputs * precision * circuit_pbs_decomposition_parameter.level) as f64
                    * complexity_pbs;

            let complexity_circuit_bs = complexity_all_pbs + complexity_all_ppks;

            if complexity_bit_extract_wo_ks + complexity_circuit_bs > best_complexity {
                continue;
            }

            let noise_ggsw = base_noise_private_packing_ks + base_noise / 2.;

            // Circuit Boostrap
            let noise_hybrid_packing = noise_modulus_switching + noise_ggsw;
            if noise_hybrid_packing > variance_max {
                continue;
            }

            let noise_one_external_product_for_cmux_tree =
                noise_atomic_pattern::variance_bootstrap::<W>(
                    cmux_tree_blind_rotate_parameters,
                    ciphertext_modulus_log,
                    Variance::from_variance(noise_ggsw),
                )
                .get_variance();

            // final out noise hybrid packing
            let noise_cmux_tree_blind_rotate =
                noise_one_external_product_for_cmux_tree * (precision * n_inputs) as f64;

            let noise_multisum = (2_f64.powf(2. * log_norm as f64)) * noise_cmux_tree_blind_rotate; // out noise * weights

            let noise_all_multisum = noise_multisum * (1 << (2 * (precision - 1))) as f64;

            let noise_ggsw_reencoding = noise_modulus_switching + noise_all_multisum;
            if noise_ggsw_reencoding > variance_max {
                continue;
            }

            let noise_max = noise_ggsw_reencoding.max(noise_hybrid_packing);

            // Shared by all pbs (like brs)
            for (ks_dp_index, &ks_decomposition_parameter) in KS_BL_FOR_CB.iter().enumerate() {
                let (noise_keyswitch, complexity_keyswitch) = micro_tab.key_switching[ks_dp_index];
                let noise_max = noise_max + noise_keyswitch;
                if noise_max > variance_max {
                    continue;
                }

                let complexity_all_ks = (precision * n_inputs) as f64 * complexity_keyswitch;
                let complexity_bit_extract = complexity_bit_extract_wo_ks + complexity_all_ks;

                let complexity_ggsw_reencoding = complexity_bit_extract + complexity_circuit_bs;

                let complexity = complexity_ggsw_reencoding + complexity_multi_hybrid_packing;

                if complexity > best_complexity {
                    // next ks.level will be even more costly
                    break;
                }

                if complexity < best_complexity {
                    let kappa = consts.kappa;
                    best_complexity = complexity;
                    let p_error = find_p_error(kappa, variance_max, noise_max);
                    let input_lwe_dimension = glwe_params.sample_extract_lwe_dimension();
                    let glwe_polynomial_size = glwe_params.polynomial_size();
                    let glwe_dimension = glwe_params.glwe_dimension;
                    state.best_solution = Some(Solution {
                        input_lwe_dimension,
                        internal_ks_output_lwe_dimension: internal_dim,
                        ks_decomposition_level_count: ks_decomposition_parameter.level,
                        ks_decomposition_base_log: ks_decomposition_parameter.log2_base,
                        glwe_polynomial_size,
                        glwe_dimension,
                        br_decomposition_level_count: br_decomposition_parameter.level,
                        br_decomposition_base_log: br_decomposition_parameter.log2_base,
                        noise_max,
                        complexity,
                        p_error,
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

const BITS_PADDING_WITHOUT_NOISE: u64 = 1;

#[allow(clippy::expect_fun_call)]
#[allow(clippy::identity_op)]
#[allow(clippy::too_many_lines)]
pub fn optimise_one<W: UnsignedInteger>(
    max_word_precision: u64, // max precision of a word
    log_norm: f64,           // ?? norm2 of noise multisum, complexity of multisum is neglected
    security_level: u64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
    n_functions: u64, // Many functions at the same time, stay at 1 for start
    n_inputs: u64,    // Tau (nb blocks)
) -> OptimizationState {
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
    let consts = OptimizationDecompositionsConsts {
        kappa,
        sum_size: 1, // Ignored
        security_level,
        noise_factor: log_norm.exp2(),
        ciphertext_modulus_log,
        keyswitch_decompositions: vec![],    // to be used later
        blind_rotate_decompositions: vec![], // to be used later
        safe_variance: variance_max,
    };

    for &glwe_dim in glwe_dimensions {
        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
            let input_lwe_dimension = glwe_dim << glwe_log_poly_size;
            // Manual experimental CUT
            if input_lwe_dimension > 1 << 13 {
                continue;
            }

            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };

            for &internal_dim in internal_lwe_dimensions {
                update_state_with_best_decompositions::<W>(
                    &mut state,
                    &consts,
                    glwe_params,
                    internal_dim,
                    n_functions,
                    max_word_precision,
                    n_inputs,
                );
            }
        }
    }

    state
}

// Default heuristic to split in several word
pub fn default_partitionning(precision: u64) -> Vec<u64> {
    #[allow(clippy::match_same_arms)]
    match precision {
        1 => vec![1],
        2 => vec![2],
        3 => vec![2; 2],
        4 => vec![3; 2],
        5 => vec![3; 2],
        6 => vec![3; 3],
        7 => vec![3; 3],
        8 => vec![3; 3],
        9 => vec![4; 3],
        10 => vec![4; 3],
        11 => vec![4; 3],
        12 => vec![4; 4],
        13 => vec![4; 4],
        14 => vec![4; 4],
        15 => vec![4; 4],
        16 => vec![5; 4],
        _ => vec![5; (precision / 5) as usize],
    }
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
) -> atomic_pattern::OptimizationState {
    let partitionning = default_partitionning(precision);
    let nb_words = partitionning.len() as u64;
    let max_word_precision = *partitionning.iter().max().unwrap() as u64;
    let log_norm = noise_factor.log2();
    let n_functions = 1;
    let result = optimise_one::<W>(
        max_word_precision,
        log_norm,
        security_level,
        maximum_acceptable_error_probability,
        glwe_log_polynomial_sizes,
        glwe_dimensions,
        internal_lwe_dimensions,
        n_functions,
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
        noise_max: sol.noise_max,
        p_error: sol.p_error,
    });
    atomic_pattern::OptimizationState {
        best_solution,
        count_domain: result.count_domain,
    }
}
