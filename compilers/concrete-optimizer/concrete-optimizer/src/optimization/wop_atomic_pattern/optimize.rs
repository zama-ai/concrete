use concrete_cpu_noise_model::gaussian_noise::conversion::variance_to_std_dev;
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;

use super::crt_decomposition;
use crate::dag::operator::Precision;
use crate::noise_estimator::error::{
    error_probability_of_sigma_scale, safe_variance_bound_product_1padbit,
    sigma_scale_of_error_probability,
};
use crate::noise_estimator::p_error::repeat_p_error;
use crate::optimization::atomic_pattern;
use crate::optimization::atomic_pattern::OptimizationDecompositionsConsts;

use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::dag::multi_parameters::keys_spec;
use crate::optimization::decomposition::circuit_bootstrap::{CbComplexityNoise, CbPareto};
use crate::optimization::decomposition::cmux::CmuxComplexityNoise;
use crate::optimization::decomposition::keyswitch::KsComplexityNoise;
use crate::optimization::decomposition::pp_switch::PpSwitchComplexityNoise;
use crate::optimization::decomposition::PersistDecompCaches;
use crate::parameters::{BrDecompositionParameters, GlweParameters};
use crate::utils::f64::f64_max;
use crate::utils::square;

pub fn find_p_error(kappa: f64, variance_bound: f64, current_maximum_noise: f64) -> f64 {
    let sigma = variance_to_std_dev(variance_bound) * kappa;
    let sigma_scale = sigma / variance_to_std_dev(current_maximum_noise);
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
    pub pp_decomposition_level_count: u64,
    pub pp_decomposition_base_log: u64,
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
            pp_decomposition_level_count: 0,
            pp_decomposition_base_log: 0,
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

fn estimate_variance(
    br_variance: f64,
    pp_variance: f64,
    cb_decomp: &CbComplexityNoise,
    ks_variance: f64,
    variance_modulus_switching: f64,
    norm: f64,
    precisions_sum: u64,
    max_precision: u64,
) -> f64 {
    assert!(max_precision <= precisions_sum);
    let variance_ggsw = pp_variance + br_variance / 2.;
    let variance_coeff_1_cmux_tree =
        square(norm)            // variance_coeff for the multisum
        * (precisions_sum                    // for hybrid packing
        << (2 * (max_precision - 1))) as f64 // for left shift
    ;
    let variance_one_external_product_for_cmux_tree = cb_decomp.variance_from_ggsw(variance_ggsw);
    variance_modulus_switching
        + variance_coeff_1_cmux_tree * variance_one_external_product_for_cmux_tree
        + ks_variance
}

fn estimate_complexity(
    glwe_params: &GlweParameters,
    br_cost: f64,
    pp_cost: f64,
    cb_decomp: &CbComplexityNoise,
    ks_cost: f64,
    precisions_sum: u64,
    nb_blocks: u64,
    n_functions: u64,
) -> f64 {
    // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
    // Hybrid packing
    let cb_level = cb_decomp.decomp.level;
    let complexity_1_cmux_hp = cb_decomp.complexity_one_cmux_hp;
    let complexity_1_ggsw_to_fft = cb_decomp.complexity_one_ggsw_to_fft;
    // BitExtract use br
    let complexity_bit_extract_1_pbs = br_cost;

    let complexity_bit_extract =
        // Assuming the last one is not done, this lets the noise goes to the CB and adds a constraint
        (precisions_sum - nb_blocks) as f64 * (complexity_bit_extract_1_pbs + ks_cost);

    // Hybrid packing
    // Circuit bs: fp-ks
    let complexity_ppks = pp_cost;
    let complexity_all_ppks =
        ((glwe_params.glwe_dimension + 1) * cb_level * precisions_sum) as f64 * complexity_ppks;

    // Circuit bs: pbs
    let complexity_cbs_ks = precisions_sum as f64 * ks_cost;
    let complexity_cbs_pbs = (precisions_sum * cb_level) as f64 * br_cost;

    let complexity_circuit_bs = complexity_cbs_ks + complexity_cbs_pbs + complexity_all_ppks;

    // Hybrid packing (Do we have 1 or 2 groups)
    let log2_polynomial_size = glwe_params.log2_polynomial_size;
    let hybrid_packing_blind_rotate_bits = log2_polynomial_size.min(precisions_sum);

    let hybrid_packing_cmux_tree_bits = precisions_sum - hybrid_packing_blind_rotate_bits;
    // Size of cmux_group, can be zero
    let hybrid_packing_cmux_tree_size = 2f64.powi(hybrid_packing_cmux_tree_bits as i32) - 1.0;
    let hybrid_packing_blind_rotate_tree_size =
        2f64.powi(hybrid_packing_blind_rotate_bits as i32) - 1.0;
    let complexity_all_ggsw_to_fft = precisions_sum as f64 * complexity_1_ggsw_to_fft;

    // Hybrid packing cmux tree
    let complexity_cmux_tree = complexity_1_cmux_hp * hybrid_packing_cmux_tree_size;

    // Hybrid packing blind rotate
    let complexity_g_br = complexity_1_cmux_hp * hybrid_packing_blind_rotate_tree_size;

    let complexity_hybrid_packing = complexity_cmux_tree + complexity_g_br;

    let complexity_multi_hybrid_packing =
        n_functions as f64 * complexity_hybrid_packing + complexity_all_ggsw_to_fft;

    complexity_bit_extract + complexity_circuit_bs + complexity_multi_hybrid_packing
}

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    glwe_params: GlweParameters,
    internal_dim: u64,
    n_functions: u64,
    precisions: &[u64],
    pareto_cmux: &[CmuxComplexityNoise],
    pareto_keyswitch: &[KsComplexityNoise],
    pp_switch: &[PpSwitchComplexityNoise],
    pareto_cb: &CbPareto,
) {
    let ciphertext_modulus_log = consts.config.ciphertext_modulus_log;
    let precisions_sum = precisions.iter().sum();
    let max_precision = *precisions.iter().max().unwrap();
    let nb_blocks = precisions.len() as u64;

    let input_lwe_dimension = glwe_params.sample_extract_lwe_dimension();

    let safe_variance_bound = consts.safe_variance;
    let norm = consts.noise_factor;

    let variance_modulus_switching = estimate_modulus_switching_noise_with_binary_key(
        internal_dim,
        glwe_params.log2_polynomial_size,
        ciphertext_modulus_log,
    );

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

    let lower_bound_variance_br = pareto_cmux.last().unwrap().noise_br(internal_dim);
    let lower_bound_variance_ks = pareto_keyswitch.last().unwrap().noise(input_lwe_dimension);
    let lower_bound_variance_private_packing = pp_switch.last().unwrap().noise;
    let lower_bound_cost_br = pareto_cmux[0].complexity_br(internal_dim);
    let lower_bound_cost_ks = pareto_keyswitch[0].complexity(input_lwe_dimension);
    let lower_bound_cost_pp = pp_switch[0].complexity;
    let lower_bound_cb = CbComplexityNoise {
        decomp: BrDecompositionParameters {
            level: 1,
            log2_base: 1,
        },
        complexity_one_cmux_hp: pareto_cb.lower_bound_cost_cb_complexity_1_cmux_hp,
        complexity_one_ggsw_to_fft: pareto_cb.lower_bound_cost_cb_complexity_1_ggsw_to_fft,
        variance_bias: pareto_cb.lower_pareto_cb_bias,
        variance_ggsw_factor: pareto_cb.lower_pareto_cb_slope,
    };

    let variance = |cmux_quantity: Option<CmuxComplexityNoise>,
                    pp_variance: Option<_>,
                    cb_quantity: Option<&CbComplexityNoise>,
                    ks_quantity: Option<KsComplexityNoise>| {
        let br_variance = cmux_quantity.map_or(lower_bound_variance_br, |quantity| {
            quantity.noise_br(internal_dim)
        });
        let pp_variance = pp_variance.unwrap_or(lower_bound_variance_private_packing);
        let cb_decomp = cb_quantity.unwrap_or(&lower_bound_cb);
        let ks_variance = ks_quantity.map_or(lower_bound_variance_ks, |quantity| {
            quantity.noise(input_lwe_dimension)
        });

        estimate_variance(
            br_variance,
            pp_variance,
            cb_decomp,
            ks_variance,
            variance_modulus_switching,
            norm,
            precisions_sum,
            max_precision,
        )
    };

    let lower_bound_variance = variance(None, None, None, None);
    if lower_bound_variance > consts.safe_variance {
        // saves 20%
        return;
    }

    let complexity = |cmux_quantity: Option<CmuxComplexityNoise>,
                      pp_cost: Option<_>,
                      cb_decomp: Option<&CbComplexityNoise>,
                      ks_quantity: Option<KsComplexityNoise>| {
        // Pbs dans BitExtract et Circuit BS et FP-KS (partagés)
        let br_cost = cmux_quantity.map_or(lower_bound_cost_br, |quantity| {
            quantity.complexity_br(internal_dim)
        });
        let ks_cost = ks_quantity.map_or(lower_bound_cost_ks, |pareto| {
            pareto.complexity(input_lwe_dimension)
        });
        let pp_cost = pp_cost.unwrap_or(lower_bound_cost_pp);
        let cb_decomp = cb_decomp.unwrap_or(&lower_bound_cb);
        estimate_complexity(
            &glwe_params,
            br_cost,
            pp_cost,
            cb_decomp,
            ks_cost,
            precisions_sum,
            nb_blocks,
            n_functions,
        )
    };

    // BlindRotate dans Circuit BS
    for &cmux_decomp in pareto_cmux {
        let lower_bound_variance = variance(Some(cmux_decomp), None, None, None);

        if lower_bound_variance > consts.safe_variance {
            // saves 20%
            continue;
        }

        // Circuit Boostrap
        // private packing keyswitch, <=> FP-KS
        for pp_switching in pp_switch {
            let lower_bound_variance =
                variance(Some(cmux_decomp), Some(pp_switching.noise), None, None);
            if lower_bound_variance > safe_variance_bound {
                continue;
            }
            let lower_bound_complexity =
                complexity(Some(cmux_decomp), Some(pp_switching.complexity), None, None);
            if lower_bound_complexity > best_complexity {
                // saves ?? TODO
                // next br_decomp are at least as costly
                break;
            }

            // CircuitBootstrap: new parameters l,b
            // for &circuit_pbs_decomposition in pareto_circuit_pbs {
            for cb_decomp in &pareto_cb.pareto {
                let lower_bound_variance = variance(
                    Some(cmux_decomp),
                    Some(pp_switching.noise),
                    Some(cb_decomp),
                    None,
                );
                if lower_bound_variance > safe_variance_bound {
                    continue;
                }
                let lower_bound_complexity = complexity(
                    Some(cmux_decomp),
                    Some(pp_switching.complexity),
                    Some(cb_decomp),
                    None,
                );
                if lower_bound_complexity > best_complexity {
                    // saves 50%
                    // next circuit_pbs_decomposition_parameter are at least as costly
                    break;
                }
                // Shared by all pbs (like brs)
                for &ks_decomp in pareto_keyswitch.iter().rev() {
                    let variance_max = variance(
                        Some(cmux_decomp),
                        Some(pp_switching.noise),
                        Some(cb_decomp),
                        Some(ks_decomp),
                    );
                    if variance_max > safe_variance_bound {
                        // saves 40%
                        break;
                    }

                    let complexity = complexity(
                        Some(cmux_decomp),
                        Some(pp_switching.complexity),
                        Some(cb_decomp),
                        Some(ks_decomp),
                    );

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
                    state.best_solution = Some(Solution {
                        input_lwe_dimension,
                        internal_ks_output_lwe_dimension: internal_dim,
                        ks_decomposition_level_count: ks_decomp.decomp.level,
                        ks_decomposition_base_log: ks_decomp.decomp.log2_base,
                        glwe_polynomial_size: glwe_params.polynomial_size(),
                        glwe_dimension: glwe_params.glwe_dimension,
                        br_decomposition_level_count: cmux_decomp.decomp.level,
                        br_decomposition_base_log: cmux_decomp.decomp.log2_base,
                        noise_max: variance_max,
                        complexity,
                        p_error,
                        global_p_error: f64::NAN,
                        cb_decomposition_level_count: cb_decomp.decomp.level,
                        cb_decomposition_base_log: cb_decomp.decomp.log2_base,
                        crt_decomposition: vec![],
                        pp_decomposition_level_count: pp_switching.decomp.level,
                        pp_decomposition_base_log: pp_switching.decomp.log2_base,
                    });
                }
            }
        }
    }
}

fn optimize_raw(
    log_norm: f64, // ?? norm2 of noise multisum, complexity of multisum is neglected
    config: Config,
    search_space: &SearchSpace,
    n_functions: u64, // Many functions at the same time, stay at 1 for start
    coprimes: &[u64],
    persistent_caches: &PersistDecompCaches,
) -> OptimizationState {
    let fractionnal_precisions = crt_decomposition::fractional_precisions_from_coprimes(coprimes);
    let precisions = crt_decomposition::precisions_from_coprimes(coprimes);
    assert!(0.0 < config.maximum_acceptable_error_probability);
    assert!(config.maximum_acceptable_error_probability < 1.0);
    assert!(!precisions.is_empty());

    let ciphertext_modulus_log = config.ciphertext_modulus_log;

    // Circuit BS bound
    // 1 bit of message only here =)
    // Bound for first bit extract in BitExtract (dominate others)
    let max_block_fractional_precision = f64_max(&fractionnal_precisions, 0.0);
    let safe_variance_bound = safe_variance_bound_product_1padbit(
        max_block_fractional_precision,
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

            let pareto_cmux = caches.cmux.pareto_quantities(glwe_params);

            let pareto_pp_switch = caches.pp_switch.pareto_quantities(glwe_params);

            let pareto_cb = caches.cb_pbs.pareto_quantities(glwe_params);

            for &internal_dim in &search_space.internal_lwe_dimensions {
                let pareto_keyswitch = caches.keyswitch.pareto_quantities(internal_dim);

                update_state_with_best_decompositions(
                    &mut state,
                    &consts,
                    glwe_params,
                    internal_dim,
                    n_functions,
                    &precisions,
                    pareto_cmux,
                    pareto_keyswitch,
                    pareto_pp_switch,
                    pareto_cb,
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
    let Ok(coprimes) = crt_decomposition::default_coprimes(precision as Precision) else {
        return OptimizationState {
            best_solution: None,
        };
    };
    let n_functions = 1;
    let mut state = optimize_raw(
        log_norm,
        config,
        search_space,
        n_functions,
        &coprimes,
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

pub fn optimize_to_circuit_solution(
    precision: u64,
    nb_instr: usize,
    nb_luts: u64,
    config: Config,
    log_norm: f64,
    search_space: &SearchSpace,
    caches: &PersistDecompCaches,
) -> keys_spec::CircuitSolution {
    let Ok(coprimes) = crt_decomposition::default_coprimes(precision as Precision) else {
        return keys_spec::CircuitSolution::no_solution(
            "Crt decomposition is not unknown for {precision}:bits",
        );
    };
    let n_functions = 1;
    let state = optimize_raw(
        log_norm,
        config,
        search_space,
        n_functions,
        &coprimes,
        caches,
    );
    if let Some(sol) = state.best_solution {
        keys_spec::CircuitSolution {
            crt_decomposition: coprimes,
            global_p_error: repeat_p_error(sol.p_error, nb_luts),
            complexity: nb_luts as f64 * sol.complexity,
            ..keys_spec::CircuitSolution::from_wop_solution(sol, nb_instr)
        }
    } else {
        keys_spec::CircuitSolution {
            crt_decomposition: coprimes,
            ..keys_spec::CircuitSolution::no_solution("No crypto parameters")
        }
    }
}
