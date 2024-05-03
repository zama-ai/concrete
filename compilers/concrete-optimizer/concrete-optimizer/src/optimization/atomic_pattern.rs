use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;

use super::config::{Config, SearchSpace};
use super::decomposition::cmux::CmuxComplexityNoise;
use super::decomposition::keyswitch::KsComplexityNoise;
use super::wop_atomic_pattern::optimize::find_p_error;
use crate::noise_estimator::error;
use crate::parameters::{BrDecompositionParameters, GlweParameters, KsDecompositionParameters};
use crate::utils::square;

use super::decomposition::{circuit_bootstrap, cmux, keyswitch, pp_switch, PersistDecompCaches};

// Ref time for v0 table 1 thread: 950ms
const CUTS: bool = true; // 80ms

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
    pub cmux: cmux::Cache,
    pub keyswitch: keyswitch::Cache,
    pub pp_switch: pp_switch::Cache,
    pub cb_pbs: circuit_bootstrap::Cache,
}

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    cmux_quantities: &[CmuxComplexityNoise],
    ks_quantities: &[KsComplexityNoise],
) {
    let input_lwe_dimension = glwe_params.sample_extract_lwe_dimension();
    let noise_modulus_switching = estimate_modulus_switching_noise_with_binary_key(
        internal_dim,
        glwe_params.log2_polynomial_size,
        consts.config.ciphertext_modulus_log,
    );
    let safe_variance = consts.safe_variance;
    if CUTS && noise_modulus_switching > safe_variance {
        return;
    }

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let mut best_variance = state.best_solution.map_or(f64::INFINITY, |s| s.noise_max);

    let complexity_multisum = (consts.sum_size * input_lwe_dimension) as f64;

    let square_noise_factor = square(consts.noise_factor);
    for cmux_quantity in cmux_quantities {
        // increasing complexity, decreasing variance
        let noise_in = cmux_quantity.noise_br(internal_dim) * square_noise_factor;
        let noise_max = noise_in + noise_modulus_switching;
        if noise_max > safe_variance && CUTS {
            continue;
        }
        let complexity_pbs = cmux_quantity.complexity_br(internal_dim);
        let complexity = complexity_multisum + complexity_pbs;
        if complexity > best_complexity {
            // As best can evolves it is complementary to blind_rotate_quantities cuts.
            break;
        }
        for &ks_quantity in ks_quantities.iter().rev() {
            let complexity_keyswitch = ks_quantity.complexity(input_lwe_dimension);
            let complexity = complexity_multisum + complexity_keyswitch + complexity_pbs;
            if complexity > best_complexity {
                continue;
            }
            // increasing variance, decreasing complexity
            let noise_keyswitch = ks_quantity.noise(input_lwe_dimension);
            let noise_max = noise_in + noise_keyswitch + noise_modulus_switching;
            if noise_max > safe_variance {
                // increasing variance => we can skip all remaining
                break;
            }
            // feasible and at least as good complexity
            if complexity < best_complexity || noise_max < best_variance {
                let p_error = find_p_error(consts.kappa, safe_variance, noise_max);

                let BrDecompositionParameters {
                    level: br_l,
                    log2_base: br_b,
                } = cmux_quantity.decomp;
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
    persistent_caches: &PersistDecompCaches,
) -> OptimizationState {
    assert!(0 < precision);
    assert!(1.0 <= noise_factor);
    assert!(0.0 < config.maximum_acceptable_error_probability);
    assert!(config.maximum_acceptable_error_probability < 1.0);

    // this assumed the noise level is equal at input/output
    // the security of the noise level of output is controlled by
    // the blind rotate decomposition

    let ciphertext_modulus_log = config.ciphertext_modulus_log;
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
    let lower_bound_cut = |glwe_log_poly_size| {
        // TODO: cut if min complexity is higher than current best
        CUTS && estimate_modulus_switching_noise_with_binary_key(
            min_internal_lwe_dimensions,
            glwe_log_poly_size,
            ciphertext_modulus_log,
        ) > consts.safe_variance
    };

    let mut caches = persistent_caches.caches();

    for glwe_params in search_space.clone().get_glwe_params() {
        assert!(8 <= glwe_params.log2_polynomial_size);
        assert!(glwe_params.log2_polynomial_size < 18);
        if lower_bound_cut(glwe_params.log2_polynomial_size) {
            continue;
        }

        let cmux_quantities = caches.cmux.pareto_quantities(glwe_params);

        for &internal_dim in &search_space.internal_lwe_dimensions {
            assert!(256 < internal_dim);

            let ks_quantities = caches.keyswitch.pareto_quantities(internal_dim);

            update_state_with_best_decompositions(
                &mut state,
                &consts,
                internal_dim,
                glwe_params,
                cmux_quantities,
                ks_quantities,
            );
        }
    }

    persistent_caches.backport(caches);

    if let Some(sol) = state.best_solution {
        assert!(0.0 <= sol.p_error && sol.p_error <= 1.0);
        assert!(sol.p_error <= config.maximum_acceptable_error_probability * REL_EPSILON_PROBA);
    }

    state
}
