use crate::computing_cost::operators::atomic_pattern as complexity_atomic_pattern;
use crate::computing_cost::operators::keyswitch_lwe::KeySwitchLWEComplexity;
use crate::computing_cost::operators::pbs::PbsComplexity;
use crate::noise_estimator::error;

use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::parameters::{
    AtomicPatternParameters, BrDecompositionParameters, GlweParameters, KeyswitchParameters,
    KsDecompositionParameters, LweDimension, PbsParameters,
};
use crate::pareto;
use crate::security;
use crate::utils::square;
use complexity_atomic_pattern::{AtomicPatternComplexity, DEFAULT as DEFAULT_COMPLEXITY};
use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::UnsignedInteger;

/* enable to debug */
const CHECKS: bool = false;
/* disable to debug */
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
pub(crate) struct OptimizationDecompositionsConsts {
    pub kappa: f64,
    pub sum_size: u64,
    pub security_level: u64,
    pub noise_factor: f64,
    pub ciphertext_modulus_log: u64,
    pub keyswitch_decompositions: Vec<KsDecompositionParameters>,
    pub blind_rotate_decompositions: Vec<BrDecompositionParameters>,
    pub safe_variance: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ComplexityNoise {
    pub index: usize,
    pub complexity: f64,
    pub noise: f64,
}

impl ComplexityNoise {
    const ZERO: Self = Self {
        index: 0,
        complexity: 0.0,
        noise: 0.0,
    };
}

pub(crate) fn cutted_blind_rotate<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    cut_complexity: f64,
    cut_noise: f64,
) -> Vec<ComplexityNoise> {
    let pareto_cut = false;
    pareto_cut_blind_rotate::<W>(
        consts,
        internal_dim,
        glwe_params,
        cut_complexity,
        cut_noise,
        pareto_cut,
    )
}

pub(crate) fn pareto_blind_rotate<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    cut_complexity: f64,
    cut_noise: f64,
) -> Vec<ComplexityNoise> {
    let pareto_cut = true;
    pareto_cut_blind_rotate::<W>(
        consts,
        internal_dim,
        glwe_params,
        cut_complexity,
        cut_noise,
        pareto_cut,
    )
}

pub(crate) fn pareto_cut_blind_rotate<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    cut_complexity: f64,
    cut_noise: f64,
    pareto_cut: bool,
) -> Vec<ComplexityNoise> {
    let br_decomp_len = consts.blind_rotate_decompositions.len();
    let mut quantities = vec![ComplexityNoise::ZERO; br_decomp_len];

    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let security_level = consts.security_level;
    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut size = 0;
    for (i_br, &br_decomposition_parameter) in consts.blind_rotate_decompositions.iter().enumerate()
    {
        let pbs_parameters = PbsParameters {
            internal_lwe_dimension: LweDimension(internal_dim),
            br_decomposition_parameter,
            output_glwe_params: glwe_params,
        };

        let complexity_pbs = DEFAULT_COMPLEXITY
            .pbs
            .complexity(pbs_parameters, ciphertext_modulus_log);

        if cut_complexity < complexity_pbs && CUTS {
            break; // complexity is increasing
        }
        let base_noise = noise_atomic_pattern::variance_bootstrap::<W>(
            pbs_parameters,
            ciphertext_modulus_log,
            variance_bsk,
        );

        let noise_out = base_noise.get_variance();
        if cut_noise < noise_out && CUTS {
            continue; // noise is decreasing
        }
        if decreasing_variance < noise_out && PARETO_CUTS && pareto_cut {
            // the current case is dominated
            continue;
        }
        let delta_complexity = complexity_pbs - increasing_complexity;
        size -= if delta_complexity == 0.0 && PARETO_CUTS && pareto_cut {
            1 // the previous case is dominated
        } else {
            0
        };
        quantities[size] = ComplexityNoise {
            index: i_br,
            complexity: complexity_pbs,
            noise: noise_out,
        };
        assert!(
            0.0 <= delta_complexity,
            "blind_rotate_decompositions should be by increasing complexity"
        );
        increasing_complexity = complexity_pbs;
        decreasing_variance = noise_out;
        size += 1;
    }
    assert!(!(PARETO_CUTS && pareto_cut) || size < 64);
    quantities.truncate(size);
    quantities
}

pub(crate) fn pareto_keyswitch<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    input_dim: u64,
    internal_dim: u64,
    cut_complexity: f64,
    cut_noise: f64,
) -> Vec<ComplexityNoise> {
    let ks_decomp_len = consts.keyswitch_decompositions.len();
    let mut quantities = vec![ComplexityNoise::ZERO; ks_decomp_len];

    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let security_level = consts.security_level;
    let variance_ksk =
        noise_atomic_pattern::variance_ksk(internal_dim, ciphertext_modulus_log, security_level);
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut size = 0;
    for (i_ks, &ks_decomposition_parameter) in consts.keyswitch_decompositions.iter().enumerate() {
        let keyswitch_parameter = KeyswitchParameters {
            input_lwe_dimension: LweDimension(input_dim),
            output_lwe_dimension: LweDimension(internal_dim),
            ks_decomposition_parameter,
        };

        let complexity_keyswitch = DEFAULT_COMPLEXITY
            .ks_lwe
            .complexity(keyswitch_parameter, ciphertext_modulus_log);
        if cut_complexity < complexity_keyswitch && CUTS {
            break;
        }
        let noise_keyswitch = noise_atomic_pattern::variance_keyswitch::<W>(
            keyswitch_parameter,
            ciphertext_modulus_log,
            variance_ksk,
        )
        .get_variance();
        if cut_noise < noise_keyswitch && CUTS {
            continue; // noise is decreasing
        }
        if decreasing_variance < noise_keyswitch && PARETO_CUTS {
            // the current case is dominated
            continue;
        }
        let delta_complexity = complexity_keyswitch - increasing_complexity;
        size -= if delta_complexity == 0.0 && PARETO_CUTS {
            1
        } else {
            0
        };
        quantities[size] = ComplexityNoise {
            index: i_ks,
            complexity: complexity_keyswitch,
            noise: noise_keyswitch,
        };
        assert!(
            0.0 <= delta_complexity,
            "keyswitch_decompositions should be by increasing complexity"
        );
        increasing_complexity = complexity_keyswitch;
        decreasing_variance = noise_keyswitch;
        size += 1;
    }
    assert!(!PARETO_CUTS || size < 64);
    quantities.truncate(size);
    quantities
}

pub struct OptimizationState {
    pub best_solution: Option<Solution>,
    pub count_domain: usize,
}

#[allow(clippy::too_many_lines)]
fn update_state_with_best_decompositions<W: UnsignedInteger>(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
) {
    let glwe_poly_size = glwe_params.polynomial_size();
    let input_lwe_dimension = glwe_params.glwe_dimension * glwe_poly_size;
    let noise_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            internal_dim,
            glwe_poly_size,
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
    let br_quantities =
        pareto_blind_rotate::<W>(consts, internal_dim, glwe_params, cut_complexity, cut_noise);
    if br_quantities.is_empty() {
        return;
    }
    if PARETO_CUTS {
        cut_noise -= br_quantities[br_quantities.len() - 1].noise;
        cut_complexity -= br_quantities[0].complexity;
    }
    let ks_quantities = pareto_keyswitch::<W>(
        consts,
        input_lwe_dimension,
        internal_dim,
        cut_complexity,
        cut_noise,
    );
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

            if CHECKS {
                assert_checks::<W>(
                    consts,
                    internal_dim,
                    glwe_params,
                    input_lwe_dimension,
                    ks_quantity,
                    br_quantity,
                    noise_max,
                    complexity_multisum,
                    complexity,
                );
            }

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

                let i_br = br_quantity.index;
                let i_ks = ks_quantity.index;
                let BrDecompositionParameters {
                    level: br_l,
                    log2_base: br_b,
                } = consts.blind_rotate_decompositions[i_br];
                let KsDecompositionParameters {
                    level: ks_l,
                    log2_base: ks_b,
                } = consts.keyswitch_decompositions[i_ks];

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

// This function provides reference values with unoptimised code, until we have non regeression tests
#[allow(clippy::float_cmp)]
#[allow(clippy::too_many_lines)]
fn assert_checks<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_params: GlweParameters,
    input_lwe_dimension: u64,
    ks_c_n: ComplexityNoise,
    br_c_n: ComplexityNoise,
    noise_max: f64,
    complexity_multisum: f64,
    complexity: f64,
) {
    let i_ks = ks_c_n.index;
    let i_br = br_c_n.index;
    let noise_out = br_c_n.noise;
    let noise_keyswitch = ks_c_n.noise;
    let complexity_keyswitch = ks_c_n.complexity;
    let complexity_pbs = br_c_n.complexity;
    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let security_level = consts.security_level;

    let br_decomposition_parameter = consts.blind_rotate_decompositions[i_br];
    let ks_decomposition_parameter = consts.keyswitch_decompositions[i_ks];

    let variance_bsk =
        security::glwe::minimal_variance(glwe_params, ciphertext_modulus_log, security_level);

    let pbs_parameters = PbsParameters {
        internal_lwe_dimension: LweDimension(internal_dim),
        br_decomposition_parameter,
        output_glwe_params: glwe_params,
    };

    let base_noise_ = noise_atomic_pattern::variance_bootstrap::<W>(
        pbs_parameters,
        ciphertext_modulus_log,
        variance_bsk,
    );
    let noise_in_ = base_noise_.get_variance() * square(consts.noise_factor);

    let complexity_pbs_ = DEFAULT_COMPLEXITY
        .pbs
        .complexity(pbs_parameters, ciphertext_modulus_log);

    assert_eq!(complexity_pbs, complexity_pbs_);
    assert_eq!(noise_out * square(consts.noise_factor), noise_in_);
    let variance_ksk =
        noise_atomic_pattern::variance_ksk(internal_dim, ciphertext_modulus_log, security_level);

    let keyswitch_parameters = KeyswitchParameters {
        input_lwe_dimension: LweDimension(input_lwe_dimension),
        output_lwe_dimension: LweDimension(internal_dim),
        ks_decomposition_parameter,
    };

    let noise_keyswitch_ = noise_atomic_pattern::variance_keyswitch::<W>(
        keyswitch_parameters,
        ciphertext_modulus_log,
        variance_ksk,
    )
    .get_variance();
    let complexity_keyswitch_ = DEFAULT_COMPLEXITY
        .ks_lwe
        .complexity(keyswitch_parameters, ciphertext_modulus_log);
    assert_eq!(complexity_keyswitch, complexity_keyswitch_);
    assert_eq!(noise_keyswitch, noise_keyswitch_);

    let atomic_pattern_parameters = AtomicPatternParameters {
        input_lwe_dimension: LweDimension(input_lwe_dimension),
        ks_decomposition_parameter,
        internal_lwe_dimension: LweDimension(internal_dim),
        br_decomposition_parameter,
        output_glwe_params: glwe_params,
    };

    let check_max_noise = noise_atomic_pattern::maximal_noise::<Variance, W>(
        Variance(noise_in_),
        atomic_pattern_parameters,
        ciphertext_modulus_log,
        security_level,
    )
    .get_variance();
    assert!(f64::abs(noise_max - check_max_noise) / check_max_noise < 0.000_000_000_01);
    let check_complexity = DEFAULT_COMPLEXITY.complexity(
        consts.sum_size,
        atomic_pattern_parameters,
        ciphertext_modulus_log,
    );

    let diff_complexity = f64::abs(complexity - check_complexity) / check_complexity;
    if diff_complexity > 0.0001 {
        println!(
            "{} + {} + {} != {}",
            complexity_multisum, complexity_keyswitch, complexity_pbs, check_complexity,
        );
    }
    assert!(diff_complexity < 0.0001);
}

const REL_EPSILON_PROBA: f64 = 1.0 + 1e-8;

#[allow(clippy::too_many_lines)]
pub fn optimize_one<W: UnsignedInteger>(
    sum_size: u64,
    precision: u64,
    security_level: u64,
    noise_factor: f64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
    restart_at: Option<Solution>,
) -> OptimizationState {
    assert!(0 < precision && precision <= 16);
    assert!(1.0 <= noise_factor);
    assert!(0.0 < maximum_acceptable_error_probability);
    assert!(maximum_acceptable_error_probability < 1.0);

    // this assumed the noise level is equal at input/output
    // the security of the noise level of ouput is controlled by
    // the blind rotate decomposition

    let ciphertext_modulus_log = W::BITS as u64;
    let safe_variance = error::safe_variance_bound_2padbits(
        precision,
        ciphertext_modulus_log,
        maximum_acceptable_error_probability,
    );
    let kappa = error::sigma_scale_of_error_probability(maximum_acceptable_error_probability);

    let consts = OptimizationDecompositionsConsts {
        kappa,
        sum_size,
        security_level,
        noise_factor,
        ciphertext_modulus_log,
        keyswitch_decompositions: pareto::KS_BL
            .map(|(log2_base, level)| KsDecompositionParameters { level, log2_base })
            .to_vec(),
        blind_rotate_decompositions: pareto::BR_BL
            .map(|(log2_base, level)| BrDecompositionParameters { level, log2_base })
            .to_vec(),
        safe_variance,
    };

    let mut state = OptimizationState {
        best_solution: None,
        count_domain: glwe_dimensions.len()
            * glwe_log_polynomial_sizes.len()
            * internal_lwe_dimensions.len()
            * consts.keyswitch_decompositions.len()
            * consts.blind_rotate_decompositions.len(),
    };

    // cut only on glwe_poly_size based of modulus switching noise
    // assume this noise is increasing with lwe_intern_dim
    let min_internal_lwe_dimensions = internal_lwe_dimensions[0];
    let lower_bound_cut = |glwe_poly_size| {
        // TODO: cut if min complexity is higher than current best
        CUTS && noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            min_internal_lwe_dimensions,
            glwe_poly_size,
        )
        .get_variance()
            > consts.safe_variance
    };

    let skip = |glwe_dim, glwe_poly_size| match restart_at {
        Some(solution) => {
            (glwe_dim, glwe_poly_size) < (solution.glwe_dimension, solution.glwe_polynomial_size)
        }
        None => false,
    };

    for &glwe_dim in glwe_dimensions {
        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
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

            for &internal_dim in internal_lwe_dimensions {
                assert!(256 < internal_dim);
                update_state_with_best_decompositions::<W>(
                    &mut state,
                    &consts,
                    internal_dim,
                    glwe_params,
                );
            }
        }
    }

    if let Some(sol) = state.best_solution {
        assert!(0.0 <= sol.p_error && sol.p_error <= 1.0);
        assert!(sol.p_error <= maximum_acceptable_error_probability * REL_EPSILON_PROBA);
    }

    state
}
