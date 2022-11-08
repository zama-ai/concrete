use super::crt_decomposition;
use crate::dag::operator::Precision;
use crate::noise_estimator::error::{
    error_probability_of_sigma_scale, safe_variance_bound_1bit_1padbit,
    sigma_scale_of_error_probability,
};
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::optimization::atomic_pattern;
use crate::optimization::atomic_pattern::OptimizationDecompositionsConsts;

use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::decomposition::{DecompCaches, PersistDecompCaches};
use crate::parameters::{GlweParameters, LweDimension, PbsParameters};
use crate::utils::square;
use concrete_commons::dispersion::{DispersionParameter, Variance};

pub fn find_p_error(kappa: f64, variance_bound: f64, current_maximum_noise: f64) -> f64 {
    let sigma = Variance(variance_bound).get_standard_dev() * kappa;
    let sigma_scale = sigma / Variance(current_maximum_noise).get_standard_dev();
    error_probability_of_sigma_scale(sigma_scale)
}

#[derive(Clone, Debug)]
pub struct OptimizationState {
    pub best_solution: Option<Solution>,
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

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    glwe_params: GlweParameters,
    internal_dim: u64,
    n_functions: u64,
    partitionning: &[u64],
    caches: &mut DecompCaches,
) {
    let ciphertext_modulus_log = consts.config.ciphertext_modulus_log;
    let precisions_sum = partitionning.iter().copied().sum();
    let max_precision = partitionning.iter().copied().max().unwrap();

    let safe_variance_bound = consts.safe_variance;
    let log_norm = consts.noise_factor.log2();

    let variance_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key(
            internal_dim,
            glwe_params.polynomial_size(),
            ciphertext_modulus_log,
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

    let pareto_blind_rotate = caches
        .blind_rotate
        .pareto_quantities(glwe_params, internal_dim);

    let pareto_keyswitch = caches
        .keyswitch
        .pareto_quantities(glwe_params, internal_dim);

    let lower_bound_variance_blind_rotate = pareto_blind_rotate.last().unwrap().noise;
    let lower_bound_variance_keyswitch = pareto_keyswitch.last().unwrap().noise;
    let lower_bound_complexity_all_ks = precisions_sum as f64 * pareto_keyswitch[0].complexity;

    let variance_coeff_br = square(consts.noise_factor) / 2.0;
    let simple_variance = |br_variance: Option<_>, ks_variance: Option<_>| {
        variance_modulus_switching
            + variance_coeff_br * br_variance.unwrap_or(lower_bound_variance_blind_rotate)
            + ks_variance.unwrap_or(lower_bound_variance_keyswitch)
    };

    let lower_bound_variance = simple_variance(None, None);
    if lower_bound_variance > consts.safe_variance {
        // saves 20%
        return;
    }

    let pp_switch = caches
        .pp_switch
        .pareto_quantities(glwe_params, internal_dim);

    let pareto_cb = caches.cb_pbs.pareto_quantities(glwe_params);

    // BlindRotate dans Circuit BS
    for (br_i, shared_br_decomp) in pareto_blind_rotate.iter().enumerate() {
        if simple_variance(Some(shared_br_decomp.noise), None) > consts.safe_variance {
            // saves 20%
            continue;
        }
        // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
        let br_decomposition_parameter = shared_br_decomp.decomp;
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        // BitExtract use this pbs
        let complexity_bit_extract_1_pbs = shared_br_decomp.complexity;
        let complexity_bit_extract_wo_ks =
            (precisions_sum - partitionning.len() as u64) as f64 * complexity_bit_extract_1_pbs;

        if complexity_bit_extract_wo_ks + lower_bound_complexity_all_ks > best_complexity {
            // saves 0%
            // next br_decomp are at least as costly
            break;
        }

        // Circuit Boostrap
        // private packing keyswitch, <=> FP-KS
        let pp_switching_index = br_i;
        let base_variance_private_packing_ks = pp_switch[pp_switching_index].noise;
        let complexity_ppks = pp_switch[pp_switching_index].complexity;

        let variance_ggsw = base_variance_private_packing_ks + shared_br_decomp.noise / 2.;

        let variance_coeff_1_cmux_tree =
            2_f64.powf(2. * log_norm) // variance_coeff for the multisum
            * (precisions_sum              // for hybrid packing
            << (2 * (max_precision - 1))) as f64 // for left shift
        ;

        // CircuitBootstrap: new parameters l,b
        // for &circuit_pbs_decomposition in pareto_circuit_pbs {
        for cb_decomp in pareto_cb {
            // Hybrid packing
            let cb_level = cb_decomp.decomp.level;
            // Circuit bs: fp-ks
            let complexity_all_ppks = ((pbs_parameters.output_glwe_params.glwe_dimension + 1)
                * cb_level
                * precisions_sum) as f64
                * complexity_ppks;

            // Circuit bs: pbs
            let complexity_all_pbs =
                (precisions_sum * cb_level) as f64 * shared_br_decomp.complexity;

            let complexity_circuit_bs = complexity_all_pbs + complexity_all_ppks;

            if complexity_bit_extract_wo_ks + lower_bound_complexity_all_ks + complexity_circuit_bs
                > best_complexity
            {
                // saves 50%
                // next circuit_pbs_decomposition_parameter are at least as costly
                break;
            }

            // Hybrid packing
            let complexity_1_cmux_hp = cb_decomp.complexity_one_cmux_hp;

            // Hybrid packing (Do we have 1 or 2 groups)
            let log2_polynomial_size = pbs_parameters.output_glwe_params.log2_polynomial_size;
            // Size of cmux_group, can be zero
            let cmux_group_count = if precisions_sum > log2_polynomial_size {
                2f64.powi((precisions_sum - log2_polynomial_size - 1) as i32)
            } else {
                0.0
            };
            let complexity_cmux_tree = cmux_group_count * complexity_1_cmux_hp;

            let complexity_one_ggsw_to_fft = cb_decomp.complexity_one_ggsw_to_fft;

            let complexity_all_ggsw_to_fft = precisions_sum as f64 * complexity_one_ggsw_to_fft;

            // Hybrid packing blind rotate
            let complexity_g_br = complexity_1_cmux_hp
                * u64::min(
                    pbs_parameters.output_glwe_params.log2_polynomial_size,
                    precisions_sum,
                ) as f64;

            let complexity_hybrid_packing = complexity_cmux_tree + complexity_g_br;

            let complexity_multi_hybrid_packing =
                n_functions as f64 * complexity_hybrid_packing + complexity_all_ggsw_to_fft;
            // Cutting on complexity here is counter-productive probably because complexity_multi_hybrid_packing is small

            let variance_one_external_product_for_cmux_tree =
                cb_decomp.variance_from_ggsw(variance_ggsw);

            // final out noise hybrid packing
            let variance_after_1st_bit_extract =
                variance_coeff_1_cmux_tree * variance_one_external_product_for_cmux_tree;

            let variance_wo_ks = variance_modulus_switching + variance_after_1st_bit_extract;

            // Shared by all pbs (like brs)
            for ks_decomp in pareto_keyswitch.iter().rev() {
                let variance_keyswitch = ks_decomp.noise;
                let variance_max = variance_wo_ks + variance_keyswitch;
                if variance_max > safe_variance_bound {
                    // saves 40%
                    break;
                }

                let complexity_all_ks = precisions_sum as f64 * ks_decomp.complexity;
                let complexity = complexity_bit_extract_wo_ks
                    + complexity_circuit_bs
                    + complexity_multi_hybrid_packing
                    + complexity_all_ks;

                if complexity > best_complexity {
                    continue;
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
                let ks_decomposition_parameter = ks_decomp.decomp;
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
                    cb_decomposition_level_count: cb_decomp.decomp.level,
                    cb_decomposition_base_log: cb_decomp.decomp.log2_base,
                    crt_decomposition: vec![],
                });
            }
        }
    }
}

fn optimize_raw(
    log_norm: f64, // ?? norm2 of noise multisum, complexity of multisum is neglected
    config: Config,
    search_space: &SearchSpace,
    n_functions: u64, // Many functions at the same time, stay at 1 for start
    partitionning: &[u64],
    persistent_caches: &PersistDecompCaches,
) -> OptimizationState {
    assert!(0.0 < config.maximum_acceptable_error_probability);
    assert!(config.maximum_acceptable_error_probability < 1.0);

    let ciphertext_modulus_log = config.ciphertext_modulus_log;

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
    };

    let consts = OptimizationDecompositionsConsts {
        config,
        kappa,
        sum_size: 1, // Ignored
        noise_factor: log_norm.exp2(),
        safe_variance: safe_variance_bound,
    };

    let mut caches = persistent_caches.caches();

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
                update_state_with_best_decompositions(
                    &mut state,
                    &consts,
                    glwe_params,
                    internal_dim,
                    n_functions,
                    partitionning,
                    &mut caches,
                );
            }
        }
    }
    persistent_caches.backport(caches);

    state
}

pub fn optimize_one(
    precision: u64,
    config: Config,
    log_norm: f64,
    search_space: &SearchSpace,
    caches: &PersistDecompCaches,
) -> OptimizationState {
    let coprimes = crt_decomposition::default_coprimes(precision as Precision);
    let partitionning = crt_decomposition::precisions_from_coprimes(&coprimes);
    let n_functions = 1;
    let mut state = optimize_raw(
        log_norm,
        config,
        search_space,
        n_functions,
        &partitionning,
        caches,
    );
    state.best_solution = state.best_solution.map(|mut sol| -> Solution {
        sol.crt_decomposition = coprimes;
        sol
    });
    state
}

pub fn optimize_one_compat(
    _sum_size: u64,
    precision: u64,
    config: Config,
    noise_factor: f64,
    search_space: &SearchSpace,
    cache: &PersistDecompCaches,
) -> atomic_pattern::OptimizationState {
    let log_norm = noise_factor.log2();
    let result = optimize_one(precision, config, log_norm, search_space, cache);
    atomic_pattern::OptimizationState {
        best_solution: result.best_solution.map(Solution::into),
    }
}
