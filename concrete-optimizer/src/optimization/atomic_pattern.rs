use super::config::{Config, SearchSpace};
use crate::noise_estimator::error;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::parameters::{BrDecompositionParameters, GlweParameters, KsDecompositionParameters};
use crate::utils::square;
use concrete_commons::dispersion::{DispersionParameter, Variance};

use super::decomposition::{blind_rotate, cut_complexity_noise, keyswitch};

// Ref time for v0 table 1 thread: 950ms
const CUTS: bool = true; // 80ms
const PARETO_CUTS: bool = true; // 75ms
const CROSS_PARETO_CUTS: bool = PARETO_CUTS && true; // 70ms

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solution {
    pub input_lwe_dimension: u64,              //n_big
    pub internal_ks_output_lwe_dimension: u64, //n_small
    pub ks_decomposition_level_count: u64,     //l(KS)
    pub ks_decomposition_base_log: u64,        //b(KS)
    pub glwe_polynomial_size: u64,             //N
    pub glwe_dimension: u64,                   //k
    pub br_decomposition_level_count: u64,     //l(BR)
    pub br_decomposition_base_log: u64,        //b(BR)
    pub complexity: f64,
    pub noise_max: f64,
    pub p_error: f64, // error probability
    pub global_p_error: f64,
}

// Constants during optimisation of decompositions
pub(crate) struct OptimizationDecompositionsConsts<'a> {
    pub config: Config<'a>,
    pub kappa: f64,
    pub sum_size: u64,
    pub noise_factor: f64,
    pub safe_variance: f64,
}

pub struct OptimizationState {
    pub best_solution: Option<Solution>,
}

pub struct Caches {
    pub blind_rotate: blind_rotate::Cache,
    pub keyswitch: keyswitch::Cache,
}

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    caches: &mut Caches,
) {
    let glwe_poly_size = glwe_params.polynomial_size();
    let input_lwe_dimension = glwe_params.glwe_dimension * glwe_poly_size;
    let noise_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key(
            internal_dim,
            glwe_poly_size,
            consts.config.ciphertext_modulus_log,
        )
        .get_variance();
    let safe_variance = consts.safe_variance;
    if CUTS && noise_modulus_switching > safe_variance {
        return;
    }

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let mut best_variance = state.best_solution.map_or(f64::INFINITY, |s| s.noise_max);

    let complexity_multisum = (consts.sum_size * input_lwe_dimension) as f64;
    let mut cut_complexity = best_complexity - complexity_multisum;
    let mut cut_noise = safe_variance - noise_modulus_switching;
    let br_quantities = caches
        .blind_rotate
        .pareto_quantities(glwe_params, internal_dim);
    let br_quantities = cut_complexity_noise(cut_complexity, cut_noise, br_quantities);
    if br_quantities.is_empty() {
        return;
    }
    if PARETO_CUTS {
        cut_noise -= br_quantities[br_quantities.len() - 1].noise;
        cut_complexity -= br_quantities[0].complexity;
    }

    let ks_quantities = caches
        .keyswitch
        .pareto_quantities(glwe_params, internal_dim);
    let ks_quantities = cut_complexity_noise(cut_complexity, cut_noise, ks_quantities);
    if ks_quantities.is_empty() {
        return;
    }

    let i_max_ks = ks_quantities.len() - 1;
    let mut i_current_max_ks = i_max_ks;
    let square_noise_factor = square(consts.noise_factor);
    for br_quantity in br_quantities {
        // increasing complexity, decreasing variance
        let noise_in = br_quantity.noise * square_noise_factor;
        let noise_max = noise_in + noise_modulus_switching;
        if noise_max > safe_variance && CUTS {
            continue;
        }
        let complexity_pbs = br_quantity.complexity;
        let complexity = complexity_multisum + complexity_pbs;
        if complexity > best_complexity {
            // As best can evolves it is complementary to blind_rotate_quantities cuts.
            if PARETO_CUTS {
                break;
            } else if CUTS {
                continue;
            }
        }
        for i_ks_pareto in (0..=i_current_max_ks).rev() {
            // increasing variance, decreasing complexity
            let ks_quantity = ks_quantities[i_ks_pareto];
            let noise_keyswitch = ks_quantity.noise;
            let noise_max = noise_in + noise_keyswitch + noise_modulus_switching;
            let complexity_keyswitch = ks_quantity.complexity;
            let complexity = complexity_multisum + complexity_keyswitch + complexity_pbs;

            if noise_max > safe_variance {
                if CROSS_PARETO_CUTS {
                    // the pareto of 2 added pareto is scanned linearly
                    // but with all cuts, pre-computing => no gain
                    i_current_max_ks = usize::min(i_ks_pareto + 1, i_max_ks);
                    break;
                    // it's compatible with next i_br but with the worst complexity
                } else if PARETO_CUTS {
                    // increasing variance => we can skip all remaining
                    break;
                }
                continue;
            } else if complexity > best_complexity {
                continue;
            }

            // feasible and at least as good complexity
            if complexity < best_complexity || noise_max < best_variance {
                let sigma = Variance(safe_variance).get_standard_dev() * consts.kappa;
                let sigma_scale = sigma / Variance(noise_max).get_standard_dev();
                let p_error = error::error_probability_of_sigma_scale(sigma_scale);

                let BrDecompositionParameters {
                    level: br_l,
                    log2_base: br_b,
                } = br_quantity.decomp;
                let KsDecompositionParameters {
                    level: ks_l,
                    log2_base: ks_b,
                } = ks_quantity.decomp;

                best_complexity = complexity;
                best_variance = noise_max;
                state.best_solution = Some(Solution {
                    input_lwe_dimension,
                    internal_ks_output_lwe_dimension: internal_dim,
                    ks_decomposition_level_count: ks_l,
                    ks_decomposition_base_log: ks_b,
                    glwe_polynomial_size: glwe_params.polynomial_size(),
                    glwe_dimension: glwe_params.glwe_dimension,
                    br_decomposition_level_count: br_l,
                    br_decomposition_base_log: br_b,
                    noise_max,
                    complexity,
                    p_error,
                    global_p_error: f64::NAN,
                });
            }
        }
    } // br ks
}

const REL_EPSILON_PROBA: f64 = 1.0 + 1e-8;

pub fn optimize_one(
    sum_size: u64,
    precision: u64,
    config: Config,
    noise_factor: f64,
    search_space: &SearchSpace,
    restart_at: Option<Solution>,
) -> OptimizationState {
    assert!(0 < precision && precision <= 16);
    assert!(1.0 <= noise_factor);
    assert!(0.0 < config.maximum_acceptable_error_probability);
    assert!(config.maximum_acceptable_error_probability < 1.0);

    // this assumed the noise level is equal at input/output
    // the security of the noise level of ouput is controlled by
    // the blind rotate decomposition

    let ciphertext_modulus_log = config.ciphertext_modulus_log;
    let security_level = config.security_level;
    let safe_variance = error::safe_variance_bound_2padbits(
        precision,
        ciphertext_modulus_log,
        config.maximum_acceptable_error_probability,
    );
    let kappa =
        error::sigma_scale_of_error_probability(config.maximum_acceptable_error_probability);

    let consts = OptimizationDecompositionsConsts {
        config,
        kappa,
        sum_size,
        noise_factor,
        safe_variance,
    };

    let mut state = OptimizationState {
        best_solution: None,
    };

    // cut only on glwe_poly_size based of modulus switching noise
    // assume this noise is increasing with lwe_intern_dim
    let min_internal_lwe_dimensions = search_space.internal_lwe_dimensions[0];
    let lower_bound_cut = |glwe_poly_size| {
        // TODO: cut if min complexity is higher than current best
        CUTS && noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key(
            min_internal_lwe_dimensions,
            glwe_poly_size,
            ciphertext_modulus_log,
        )
        .get_variance()
            > consts.safe_variance
    };

    let skip = |glwe_dim, glwe_poly_size| match restart_at {
        Some(solution) => {
            glwe_dim < solution.glwe_dimension && glwe_poly_size < solution.glwe_polynomial_size
        }
        None => false,
    };

    let mut caches = Caches {
        blind_rotate: blind_rotate::for_security(security_level).cache(),
        keyswitch: keyswitch::for_security(security_level).cache(),
    };

    for &glwe_dim in &search_space.glwe_dimensions {
        for &glwe_log_poly_size in &search_space.glwe_log_polynomial_sizes {
            assert!(8 <= glwe_log_poly_size);
            assert!(glwe_log_poly_size < 18);
            let glwe_poly_size = 1 << glwe_log_poly_size;
            if lower_bound_cut(glwe_poly_size) {
                continue;
            }
            if skip(glwe_dim, glwe_poly_size) {
                continue;
            }

            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };

            for &internal_dim in &search_space.internal_lwe_dimensions {
                assert!(256 < internal_dim);
                update_state_with_best_decompositions(
                    &mut state,
                    &consts,
                    internal_dim,
                    glwe_params,
                    &mut caches,
                );
            }
        }
    }

    blind_rotate::for_security(security_level).backport(caches.blind_rotate);
    keyswitch::for_security(security_level).backport(caches.keyswitch);

    if let Some(sol) = state.best_solution {
        assert!(0.0 <= sol.p_error && sol.p_error <= 1.0);
        assert!(sol.p_error <= config.maximum_acceptable_error_probability * REL_EPSILON_PROBA);
    }

    state
}
