use super::crt_decomposition;
use crate::computing_cost::complexity::Complexity;
use crate::computing_cost::operators::cmux;
use crate::dag::operator::Precision;
use crate::noise_estimator::error::{
    error_probability_of_sigma_scale, safe_variance_bound_1bit_1padbit,
    sigma_scale_of_error_probability,
};
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::noise_estimator::operators::wop_atomic_pattern::estimate_packing_private_keyswitch;
use crate::optimization::atomic_pattern;
use crate::optimization::atomic_pattern::{
    cutted_blind_rotate, pareto_keyswitch, ComplexityNoise, OptimizationDecompositionsConsts,
};
use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::wop_atomic_pattern::pareto::{
    BR_CIRCUIT_BOOTSTRAP_PARETO_DECOMP, BR_PARETO_DECOMP, KS_CIRCUIT_BOOTSTRAP_PARETO_DECOMP,
    KS_PARETO_DECOMP,
};
use crate::parameters::{
    GlweParameters, KeyswitchParameters, KsDecompositionParameters, LweDimension, PbsParameters,
};
use crate::security;
use crate::utils::square;
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::UnsignedInteger;

pub fn find_p_error(kappa: f64, variance_bound: f64, current_maximum_noise: f64) -> f64 {
    let sigma = Variance(variance_bound).get_standard_dev() * kappa;
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
    pub global_p_error: f64,
    // error probability
    pub cb_decomposition_level_count: u64,
    pub cb_decomposition_base_log: u64,
    pub crt_decomposition: Vec<u64>,
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
            global_p_error: 0.0,
            cb_decomposition_level_count: 0,
            cb_decomposition_base_log: 0,
            crt_decomposition: vec![],
        }
    }
}

impl From<Solution> for atomic_pattern::Solution {
    fn from(sol: Solution) -> Self {
        Self {
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
            global_p_error: sol.global_p_error,
        }
    }
}

#[derive(Debug)]
struct NoiseCostByMicroParam {
    cutted_blind_rotate: Vec<ComplexityNoise>,
    pareto_keyswitch: Vec<ComplexityNoise>,
    pp_switching: Vec<(f64, Complexity)>,
}

fn compute_noise_cost_by_micro_param<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    glwe_params: GlweParameters,
    internal_dim: u64,
    best_complexity: f64,
    variance_modulus_switching: f64,
    precision: u64,
    n_inputs: u64,
) -> Option<NoiseCostByMicroParam> {
    let security_level = consts.config.security_level;

    let variance_coeff = square(consts.noise_factor) / 2.0;
    let complexity_coeff = (n_inputs * (2 * precision - 1)) as f64;
    let cut_complexity = best_complexity / complexity_coeff; // saves 0%
    let cut_variance = (consts.safe_variance - variance_modulus_switching) / variance_coeff; // saves 40%

    let cutted_blind_rotate = cutted_blind_rotate::<W>(
        consts,
        internal_dim,
        glwe_params,
        cut_complexity,
        cut_variance,
    );
    if cutted_blind_rotate.is_empty() {
        return None;
    }

    let variance_coeff_br = variance_coeff;
    let variance_coeff = 1.0;
    let complexity_coeff = (precision * n_inputs) as f64;
    let cut_complexity = best_complexity / complexity_coeff; // saves 0%
    let cut_variance = (consts.safe_variance
        - variance_modulus_switching
        - variance_coeff_br * cutted_blind_rotate.last().unwrap().noise)
        / variance_coeff; // saves 25%

    let input_dim = glwe_params.sample_extract_lwe_dimension();
    let pareto_keyswitch = pareto_keyswitch::<W>(
        consts,
        input_dim,
        internal_dim,
        cut_complexity,
        cut_variance,
    );
    if pareto_keyswitch.is_empty() {
        return None;
    }

    let ciphertext_modulus_log = W::BITS as u64;

    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);

    let mut variance_cost_pp_switching = vec![(f64::NAN, f64::NAN); BR_PARETO_DECOMP.len()];
    for br in &cutted_blind_rotate {
        // saves 0%
        let pp_ks_decomposition_parameter = BR_PARETO_DECOMP[br.index];
        let ppks_parameter = PbsParameters {
            internal_lwe_dimension: LweDimension(
                glwe_params.glwe_dimension * glwe_params.polynomial_size(),
            ),
            br_decomposition_parameter: pp_ks_decomposition_parameter,
            output_glwe_params: glwe_params,
        };
        // We assume the packing KS and the external product in a PBSto have
        // the same parameters (base, level)
        let variance_private_packing_ks =
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
        let complexity_ppks = consts
            .config
            .complexity_model
            .ks_complexity(ppks_parameter_complexity, ciphertext_modulus_log);
        variance_cost_pp_switching[br.index] = (variance_private_packing_ks, complexity_ppks);
    }

    Some(NoiseCostByMicroParam {
        cutted_blind_rotate,
        pareto_keyswitch,
        pp_switching: variance_cost_pp_switching,
    })
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
    let ciphertext_modulus_log = consts.config.ciphertext_modulus_log;
    let global_precision = n_inputs * precision;
    let safe_variance_bound = consts.safe_variance;
    let log_norm = consts.noise_factor.log2();

    let variance_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            internal_dim,
            glwe_params.polynomial_size(),
        )
        .get_variance();

    if variance_modulus_switching > consts.safe_variance {
        return;
    }

    let mut best_complexity = state
        .best_solution
        .as_ref()
        .map_or(f64::INFINITY, |s| s.complexity);
    let mut best_variance = state
        .best_solution
        .as_ref()
        .map_or(f64::INFINITY, |s| s.noise_max);

    let variance_cost_opt = compute_noise_cost_by_micro_param::<W>(
        consts,
        glwe_params,
        internal_dim,
        best_complexity,
        variance_modulus_switching,
        precision,
        n_inputs,
    );
    let variance_cost = if let Some(variance_cost) = variance_cost_opt {
        variance_cost
    } else {
        return;
    };

    // pareto keyswitch is sorted by complexity increasing and variance decreasing
    let lower_bound_variance_keyswitch =
        variance_cost.pareto_keyswitch[variance_cost.pareto_keyswitch.len() - 1].noise;
    let lower_bound_complexity_all_ks =
        (precision * n_inputs) as f64 * variance_cost.pareto_keyswitch[0].complexity;

    // BlindRotate dans Circuit BS
    for shared_br_decomp in &variance_cost.cutted_blind_rotate {
        // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
        let br_decomposition_parameter = consts.blind_rotate_decompositions[shared_br_decomp.index];
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        // BitExtract use this pbs
        let complexity_bit_extract_1_pbs = shared_br_decomp.complexity;
        let complexity_bit_extract_wo_ks =
            (n_inputs * (precision - 1)) as f64 * complexity_bit_extract_1_pbs;

        if complexity_bit_extract_wo_ks + lower_bound_complexity_all_ks > best_complexity {
            // saves 0%
            // next br_decomp are at least as costly
            break;
        }

        // Circuit Boostrap
        // private packing keyswitch, <=> FP-KS
        let pp_switching_index = shared_br_decomp.index;
        let (base_variance_private_packing_ks, complexity_ppks) =
            variance_cost.pp_switching[pp_switching_index];

        let variance_ggsw = base_variance_private_packing_ks + shared_br_decomp.noise / 2.;

        let variance_coeff_1_cmux_tree =
            2_f64.powf(2. * log_norm as f64) // variance_coeff for the multisum
            * (global_precision              // for hybrid packing
            << (2 * (precision - 1))) as f64 // for left shift
        ;

        // CircuitBootstrap: new parameters l,b
        for &circuit_pbs_decomposition_parameter in BR_CIRCUIT_BOOTSTRAP_PARETO_DECOMP.iter() {
            // Hybrid packing
            let nb_cmux = 1_u64;
            let cmux_tree_blind_rotate_parameters = PbsParameters {
                internal_lwe_dimension: LweDimension(nb_cmux), // complexity for 1 cmux
                br_decomposition_parameter: circuit_pbs_decomposition_parameter,
                output_glwe_params: pbs_parameters.output_glwe_params,
            };

            // Circuit bs: fp-ks
            let complexity_all_ppks = ((pbs_parameters.output_glwe_params.glwe_dimension + 1)
                * circuit_pbs_decomposition_parameter.level
                * global_precision) as f64
                * complexity_ppks;

            // Circuit bs: pbs
            let complexity_all_pbs = (global_precision * circuit_pbs_decomposition_parameter.level)
                as f64
                * shared_br_decomp.complexity;

            let complexity_circuit_bs = complexity_all_pbs + complexity_all_ppks;

            if complexity_bit_extract_wo_ks + lower_bound_complexity_all_ks + complexity_circuit_bs
                > best_complexity
            {
                // saves 50%
                // next circuit_pbs_decomposition_parameter are at least as costly
                break;
            }

            // Hybrid packing
            let complexity_1_cmux_hp = consts
                .config
                .complexity_model
                .pbs_complexity(cmux_tree_blind_rotate_parameters, ciphertext_modulus_log); // TODO: missing fft transform

            // Hybrid packing (Do we have 1 or 2 groups)
            let log2_polynomial_size = pbs_parameters.output_glwe_params.log2_polynomial_size;
            // Size of cmux_group, can be zero
            let cmux_group_count = if global_precision > log2_polynomial_size {
                2f64.powi((global_precision - log2_polynomial_size - 1) as i32)
            } else {
                0.0
            };
            let complexity_cmux_tree = cmux_group_count as f64 * complexity_1_cmux_hp;

            let cmux_complexity = cmux::SimpleWithFactors::default();

            let f_glwe_poly_size = glwe_params.polynomial_size() as f64;

            let f_glwe_size = (glwe_params.glwe_dimension + 1) as f64;

            let complexity_one_ggsw_to_fft = square(f_glwe_size)
                * circuit_pbs_decomposition_parameter.level as f64
                * cmux_complexity.fft_complexity(f_glwe_poly_size, ciphertext_modulus_log);

            let complexity_all_ggsw_to_fft =
                (1 << global_precision) as f64 * complexity_one_ggsw_to_fft;

            // Hybrid packing blind rotate
            let complexity_g_br = complexity_1_cmux_hp
                * u64::min(
                    pbs_parameters.output_glwe_params.log2_polynomial_size,
                    global_precision,
                ) as f64;

            let complexity_hybrid_packing = complexity_cmux_tree + complexity_g_br;

            let complexity_multi_hybrid_packing =
                n_functions as f64 * complexity_hybrid_packing + complexity_all_ggsw_to_fft;
            // Cutting on complexity here is counter-productive probably because complexity_multi_hybrid_packing is small

            let variance_one_external_product_for_cmux_tree =
                noise_atomic_pattern::variance_bootstrap::<W>(
                    cmux_tree_blind_rotate_parameters,
                    ciphertext_modulus_log,
                    Variance::from_variance(variance_ggsw),
                )
                .get_variance();

            // final out noise hybrid packing
            let variance_after_1st_bit_extract =
                variance_coeff_1_cmux_tree * variance_one_external_product_for_cmux_tree;

            let variance_wo_ks = variance_modulus_switching + variance_after_1st_bit_extract;

            if variance_wo_ks + lower_bound_variance_keyswitch > safe_variance_bound {
                // saves 40%
                continue;
            }

            // Shared by all pbs (like brs)
            for ks_decomp in &variance_cost.pareto_keyswitch {
                let variance_keyswitch = ks_decomp.noise;
                let variance_max = variance_wo_ks + variance_keyswitch;
                if variance_max > safe_variance_bound {
                    continue;
                }

                let complexity_all_ks = (precision * n_inputs) as f64 * ks_decomp.complexity;
                let complexity = complexity_bit_extract_wo_ks
                    + complexity_circuit_bs
                    + complexity_multi_hybrid_packing
                    + complexity_all_ks;

                if complexity > best_complexity {
                    // next ks.level will be even more costly
                    break;
                }
                #[allow(clippy::float_cmp)]
                if complexity == best_complexity && variance_max > best_variance {
                    continue;
                }

                let kappa = consts.kappa;
                best_complexity = complexity;
                best_variance = variance_max;
                let p_error = find_p_error(kappa, safe_variance_bound, variance_max);
                let input_lwe_dimension = glwe_params.sample_extract_lwe_dimension();
                let glwe_polynomial_size = glwe_params.polynomial_size();
                let glwe_dimension = glwe_params.glwe_dimension;
                let ks_decomposition_parameter = consts.keyswitch_decompositions[ks_decomp.index];
                state.best_solution = Some(Solution {
                    input_lwe_dimension,
                    internal_ks_output_lwe_dimension: internal_dim,
                    ks_decomposition_level_count: ks_decomposition_parameter.level,
                    ks_decomposition_base_log: ks_decomposition_parameter.log2_base,
                    glwe_polynomial_size,
                    glwe_dimension,
                    br_decomposition_level_count: br_decomposition_parameter.level,
                    br_decomposition_base_log: br_decomposition_parameter.log2_base,
                    noise_max: variance_max,
                    complexity,
                    p_error,
                    global_p_error: f64::NAN,
                    cb_decomposition_level_count: circuit_pbs_decomposition_parameter.level,
                    cb_decomposition_base_log: circuit_pbs_decomposition_parameter.log2_base,
                    crt_decomposition: vec![],
                });
            }
        }
    }
}

fn optimize_raw<W: UnsignedInteger>(
    max_word_precision: u64, // max precision of a word
    log_norm: f64,           // ?? norm2 of noise multisum, complexity of multisum is neglected
    config: Config,
    search_space: &SearchSpace,
    n_functions: u64, // Many functions at the same time, stay at 1 for start
    n_inputs: u64,    // Tau (nb blocks)
) -> OptimizationState {
    assert!(0.0 < config.maximum_acceptable_error_probability);
    assert!(config.maximum_acceptable_error_probability < 1.0);

    let ciphertext_modulus_log = W::BITS as u64;

    // Circuit BS bound
    // 1 bit of message only here =)
    // Bound for first bit extract in BitExtract (dominate others)
    let safe_variance_bound = safe_variance_bound_1bit_1padbit(
        ciphertext_modulus_log,
        config.maximum_acceptable_error_probability,
    );
    let kappa: f64 = sigma_scale_of_error_probability(config.maximum_acceptable_error_probability);

    let mut state = OptimizationState {
        best_solution: None,
        count_domain: search_space.glwe_dimensions.len()
            * search_space.glwe_log_polynomial_sizes.len()
            * search_space.internal_lwe_dimensions.len()
            * KS_PARETO_DECOMP.len()
            * BR_PARETO_DECOMP.len(),
    };

    let consts = OptimizationDecompositionsConsts {
        config,
        kappa,
        sum_size: 1, // Ignored
        noise_factor: log_norm.exp2(),
        keyswitch_decompositions: KS_CIRCUIT_BOOTSTRAP_PARETO_DECOMP.to_vec(),
        blind_rotate_decompositions: BR_PARETO_DECOMP.to_vec(),
        safe_variance: safe_variance_bound,
    };

    for &glwe_dim in &search_space.glwe_dimensions {
        for &glwe_log_poly_size in &search_space.glwe_log_polynomial_sizes {
            let input_lwe_dimension = glwe_dim << glwe_log_poly_size;
            // Manual experimental CUT
            if input_lwe_dimension > 1 << 13 {
                continue;
            }

            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };

            for &internal_dim in &search_space.internal_lwe_dimensions {
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

pub fn optimize_one<W: UnsignedInteger>(
    precision: u64,
    config: Config,
    log_norm: f64,
    search_space: &SearchSpace,
) -> OptimizationState {
    let coprimes = crt_decomposition::default_coprimes(precision as Precision);
    let partitionning = crt_decomposition::precisions_from_coprimes(&coprimes);
    let nb_words = partitionning.len() as u64;
    let max_word_precision = *partitionning.iter().max().unwrap() as u64;
    let n_functions = 1;
    let mut state = optimize_raw::<W>(
        max_word_precision,
        log_norm,
        config,
        search_space,
        n_functions,
        nb_words, // Tau
    );
    state.best_solution = state.best_solution.map(|mut sol| -> Solution {
        sol.crt_decomposition = coprimes;
        sol
    });
    state
}

pub fn optimize_one_compat<W: UnsignedInteger>(
    _sum_size: u64,
    precision: u64,
    config: Config,
    noise_factor: f64,
    search_space: &SearchSpace,
) -> atomic_pattern::OptimizationState {
    let log_norm = noise_factor.log2();
    let result = optimize_one::<W>(precision, config, log_norm, search_space);
    atomic_pattern::OptimizationState {
        best_solution: result.best_solution.map(Solution::into),
        count_domain: result.count_domain,
    }
}
