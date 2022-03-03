use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::UnsignedInteger;

use crate::computing_cost::operators::atomic_pattern as complexity_atomic_pattern;
use complexity_atomic_pattern::AtomicPatternComplexity;

use crate::computing_cost::operators::keyswitch_lwe::KeySwitchLWEComplexity;
use crate::computing_cost::operators::pbs::PbsComplexity;

use crate::noise_estimator::error::{
    error_probability_of_sigma_scale, sigma_scale_of_error_probability,
};
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::security;

#[rustfmt::skip]
const BR_BL: &[(u64, u64); 35] = &[
    (12, 1), (23, 1), (8, 2), (15, 2), (16, 2), (3, 3), (6, 3), (12, 3), (2, 4),
    (5, 4), (9, 4), (4, 5), (8, 5), (7, 6), (3, 7), (6, 7), (1, 8), (5, 8), (1, 9),
    (5, 9), (2, 10), (4, 10), (2, 11), (4, 11), (3, 14), (3, 15), (1, 21), (2, 21), (1, 22),
    (2, 22), (2, 23), (1, 43), (1, 44), (1, 45), (1, 46)
];

#[rustfmt::skip]
const KS_BL: &[(u64, u64); 46] = &[
    (5, 1), (12, 1), (26, 1), (31, 1), (4, 2), (8, 2), (17, 2), (21, 2), (3, 3),
    (6, 3), (13, 3), (15, 3), (2, 4), (5, 4), (10, 4), (12, 4), (2, 5), (4, 5),
    (9, 5), (10, 5), (4, 6), (8, 6), (3, 7), (7, 7), (3, 8), (6, 8), (1, 9), (5, 9), (1, 10),
    (5, 10), (2, 11), (2, 12), (4, 12), (4, 13), (3, 16), (3, 17), (1, 22), (1, 23), (2, 24),
    (2, 25), (2, 26), (1, 48), (1, 49), (1, 50), (1, 51), (1, 52)
];

fn square(v: f64) -> f64 {
    v * v
}

/* enable to debug */
const CHECKS: bool = false;
/* disable to debug */
// Ref time for v0 table 1 thread: 950ms
const CUTS: bool = true; // 80ms
const PARETO_CUTS: bool = true; // 75ms
const CROSS_PARETO_CUTS: bool = PARETO_CUTS && true; // 70ms

#[derive(Debug, Clone, Copy)]
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
}

// Constants during optimisation of decompositions
struct OptimizationDecompositionsConsts {
    kappa: f64,
    sum_size: u64,
    security_level: u64,
    noise_factor: f64,
    ciphertext_modulus_log: u64,
    keyswitch_decompositions: Vec<(u64, u64)>,
    blind_rotate_decompositions: Vec<(u64, u64)>,
    variance_max: f64,
}

#[derive(Clone, Copy)]
struct ComplexityNoise {
    index: usize,
    complexity: f64,
    noise: f64,
}

impl ComplexityNoise {
    const ZERO: Self = Self {
        index: 0,
        complexity: 0.0,
        noise: 0.0,
    };
}

fn blind_rotate_quantities<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_poly_size: u64,
    glwe_dim: u64,
    cut_complexity: f64,
    cut_noise: f64,
) -> Vec<ComplexityNoise> {
    let br_decomp_len = consts.blind_rotate_decompositions.len();
    let mut quantities = vec![ComplexityNoise::ZERO; br_decomp_len];

    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let security_level = consts.security_level;
    let variance_bsk = security::glwe::minimal_variance(
        glwe_poly_size,
        glwe_dim,
        ciphertext_modulus_log,
        security_level,
    );
    let mut increasing_complexity = 0.0;
    let mut decreasing_variance = f64::INFINITY;
    let mut size = 0;
    for (i_br, &(br_b, br_l)) in consts.blind_rotate_decompositions.iter().enumerate() {
        let complexity_pbs = complexity_atomic_pattern::DEFAULT.pbs.complexity(
            internal_dim,
            glwe_poly_size,
            glwe_dim,
            br_l,
            br_b,
            consts.ciphertext_modulus_log,
        );
        if cut_complexity < complexity_pbs && CUTS {
            break; // complexity is increasing
        }
        let base_noise = noise_atomic_pattern::variance_bootstrap::<W>(
            internal_dim,
            glwe_poly_size,
            glwe_dim,
            br_l,
            br_b,
            consts.ciphertext_modulus_log,
            variance_bsk,
        );
        let noise_in = base_noise.get_variance() * square(consts.noise_factor);
        if cut_noise < noise_in && CUTS {
            continue; // noise is decreasing
        }
        if decreasing_variance < noise_in && PARETO_CUTS {
            // the current case is dominated
            continue;
        }
        let delta_complexity = complexity_pbs - increasing_complexity;
        size -= if delta_complexity == 0.0 && PARETO_CUTS {
            1 // the previous case is dominated
        } else {
            0
        };
        quantities[size] = ComplexityNoise {
            index: i_br,
            complexity: complexity_pbs,
            noise: noise_in,
        };
        assert!(
            0.0 <= delta_complexity,
            "blind_rotate_decompositions should be by increasing complexity"
        );
        increasing_complexity = complexity_pbs;
        decreasing_variance = noise_in;
        size += 1;
    }
    assert!(!PARETO_CUTS || size < 64);
    quantities.truncate(size);
    quantities
}

fn keyswitch_quantities<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    in_dim: u64,
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
    for (i_ks, &(ks_b, ks_l)) in consts.keyswitch_decompositions.iter().enumerate() {
        let complexity_keyswitch = complexity_atomic_pattern::DEFAULT.ks_lwe.complexity(
            in_dim,
            internal_dim,
            ks_l,
            ks_b,
            ciphertext_modulus_log,
        );
        if cut_complexity < complexity_keyswitch && CUTS {
            break;
        }
        let noise_keyswitch = noise_atomic_pattern::variance_keyswitch::<W>(
            in_dim,
            ks_l,
            ks_b,
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
    glwe_poly_size: u64,
    glwe_dim: u64,
) {
    let input_lwe_dimension = glwe_dim * glwe_poly_size;
    let noise_modulus_switching =
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            internal_dim,
            glwe_poly_size,
        )
        .get_variance();
    let variance_max = consts.variance_max;
    if CUTS && noise_modulus_switching > variance_max {
        return;
    }

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let mut best_variance = state.best_solution.map_or(f64::INFINITY, |s| s.noise_max);

    let complexity_multisum = (consts.sum_size * input_lwe_dimension) as f64;
    let mut cut_complexity = best_complexity - complexity_multisum;
    let mut cut_noise = variance_max - noise_modulus_switching;
    let br_quantities = blind_rotate_quantities::<W>(
        consts,
        internal_dim,
        glwe_poly_size,
        glwe_dim,
        cut_complexity,
        cut_noise,
    );
    if br_quantities.is_empty() {
        return;
    }
    if PARETO_CUTS {
        cut_noise -= br_quantities[br_quantities.len() - 1].noise;
        cut_complexity -= br_quantities[0].complexity;
    }
    let ks_quantities = keyswitch_quantities::<W>(
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
    for br_quantity in br_quantities {
        // increasing complexity, decreasing variance
        let noise_in = br_quantity.noise;
        let noise_max = noise_in + noise_modulus_switching;
        if noise_max > variance_max && CUTS {
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
                    glwe_poly_size,
                    glwe_dim,
                    input_lwe_dimension,
                    ks_quantity,
                    br_quantity,
                    noise_max,
                    complexity_multisum,
                    complexity,
                );
            }

            if noise_max > variance_max {
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
                let sigma = Variance(variance_max).get_standard_dev() * consts.kappa;
                let sigma_scale = sigma / Variance(noise_max).get_standard_dev();
                let p_error = error_probability_of_sigma_scale(sigma_scale);

                let i_br = br_quantity.index;
                let i_ks = ks_quantity.index;
                let (br_b, br_l) = consts.blind_rotate_decompositions[i_br];
                let (ks_b, ks_l) = consts.keyswitch_decompositions[i_ks];

                best_complexity = complexity;
                best_variance = noise_max;
                state.best_solution = Some(Solution {
                    input_lwe_dimension,
                    internal_ks_output_lwe_dimension: internal_dim,
                    ks_decomposition_level_count: ks_l,
                    ks_decomposition_base_log: ks_b,
                    glwe_polynomial_size: glwe_poly_size,
                    glwe_dimension: glwe_dim,
                    br_decomposition_level_count: br_l,
                    br_decomposition_base_log: br_b,
                    noise_max,
                    complexity,
                    p_error,
                });
            }
        }
    } // br ks
}

// This function provides reference values with unoptimised code, until we have non regeression tests
#[allow(clippy::float_cmp)]
fn assert_checks<W: UnsignedInteger>(
    consts: &OptimizationDecompositionsConsts,
    internal_dim: u64,
    glwe_poly_size: u64,
    glwe_dim: u64,
    input_lwe_dimension: u64,
    ks_c_n: ComplexityNoise,
    br_c_n: ComplexityNoise,
    noise_max: f64,
    complexity_multisum: f64,
    complexity: f64,
) {
    let i_ks = ks_c_n.index;
    let i_br = br_c_n.index;
    let noise_in = br_c_n.noise;
    let noise_keyswitch = ks_c_n.noise;
    let complexity_keyswitch = ks_c_n.complexity;
    let complexity_pbs = br_c_n.complexity;
    let ciphertext_modulus_log = consts.ciphertext_modulus_log;
    let security_level = consts.security_level;
    let (br_b, br_l) = consts.blind_rotate_decompositions[i_br];
    let (ks_b, ks_l) = consts.keyswitch_decompositions[i_ks];
    let variance_bsk = security::glwe::minimal_variance(
        glwe_poly_size,
        glwe_dim,
        ciphertext_modulus_log,
        security_level,
    );
    let base_noise_ = noise_atomic_pattern::variance_bootstrap::<W>(
        internal_dim,
        glwe_poly_size,
        glwe_dim,
        br_l,
        br_b,
        ciphertext_modulus_log,
        variance_bsk,
    );
    let noise_in_ = base_noise_.get_variance() * square(consts.noise_factor);
    let complexity_pbs_ = complexity_atomic_pattern::DEFAULT.pbs.complexity(
        internal_dim,
        glwe_poly_size,
        glwe_dim,
        br_l,
        br_b,
        ciphertext_modulus_log,
    );
    assert!(complexity_pbs == complexity_pbs_);
    assert!(noise_in == noise_in_);
    let variance_ksk =
        noise_atomic_pattern::variance_ksk(internal_dim, ciphertext_modulus_log, security_level);
    let noise_keyswitch_ = noise_atomic_pattern::variance_keyswitch::<W>(
        input_lwe_dimension,
        ks_l,
        ks_b,
        ciphertext_modulus_log,
        variance_ksk,
    )
    .get_variance();
    let complexity_keyswitch_ = complexity_atomic_pattern::DEFAULT.ks_lwe.complexity(
        input_lwe_dimension,
        internal_dim,
        ks_l,
        ks_b,
        ciphertext_modulus_log,
    );
    assert!(complexity_keyswitch == complexity_keyswitch_);
    assert!(noise_keyswitch == noise_keyswitch_);

    let check_max_noise = noise_atomic_pattern::maximal_noise::<Variance, W>(
        Variance(noise_in),
        input_lwe_dimension,
        internal_dim,
        ks_l,
        ks_b,
        glwe_poly_size,
        ciphertext_modulus_log,
        security_level,
    )
    .get_variance();
    assert!(f64::abs(noise_max - check_max_noise) / check_max_noise < 0.00000000001);
    let check_complexity = complexity_atomic_pattern::DEFAULT.complexity(
        consts.sum_size,
        input_lwe_dimension,
        internal_dim,
        ks_l,
        ks_b,
        glwe_poly_size,
        glwe_dim,
        br_l,
        br_b,
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

const BITS_CARRY: u64 = 1;
const BITS_PADDING_WITHOUT_NOISE: u64 = 1;

#[allow(clippy::too_many_lines)]
pub fn optimise_one<W: UnsignedInteger>(
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
    assert!(security_level == 128);
    assert!(1.0 <= noise_factor);
    assert!(0.0 < maximum_acceptable_error_probability);
    assert!(maximum_acceptable_error_probability < 1.0);

    // this assumed the noise level is equal at input/output
    // the security of the noise level of ouput is controlled by
    // the blind rotate decomposition

    let ciphertext_modulus_log = W::BITS as u64;

    let no_noise_bits = BITS_CARRY + precision + BITS_PADDING_WITHOUT_NOISE;
    let noise_bits = ciphertext_modulus_log - no_noise_bits;
    let fatal_noise_limit = (1_u64 << noise_bits) as f64;

    // Now we search for P(x not in [-+fatal_noise_limit] | σ = safe_sigma) = p_error
    // P(x not in [-+kappa] | σ = 1) = p_error
    let kappa: f64 = sigma_scale_of_error_probability(maximum_acceptable_error_probability);
    let safe_sigma = fatal_noise_limit / kappa;
    let variance_max = Variance::from_modular_variance::<W>(square(safe_sigma));

    let consts = OptimizationDecompositionsConsts {
        kappa,
        sum_size,
        security_level,
        noise_factor,
        ciphertext_modulus_log,
        keyswitch_decompositions: KS_BL.to_vec(),
        blind_rotate_decompositions: BR_BL.to_vec(),
        variance_max: variance_max.get_variance(),
    };

    let mut state = OptimizationState {
        best_solution: None,
        count_domain: glwe_dimensions.len()
            * glwe_log_polynomial_sizes.len()
            * internal_lwe_dimensions.len()
            * KS_BL.len()
            * BR_BL.len(),
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
            > consts.variance_max
    };

    let skip = |glwe_dim, glwe_poly_size| match restart_at {
        Some(solution) => {
            (glwe_dim, glwe_poly_size) < (solution.glwe_dimension, solution.glwe_polynomial_size)
        }
        None => false,
    };

    for &glwe_dim in glwe_dimensions {
        assert!(1 <= glwe_dim);
        assert!(glwe_dim < 4);

        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
            assert!(8 < glwe_log_poly_size);
            assert!(glwe_log_poly_size < 18);
            let glwe_poly_size = 1 << glwe_log_poly_size;
            if lower_bound_cut(glwe_poly_size) {
                continue;
            }
            if skip(glwe_dim, glwe_poly_size) {
                continue;
            }

            for &internal_dim in internal_lwe_dimensions {
                assert!(256 < internal_dim);
                update_state_with_best_decompositions::<W>(
                    &mut state,
                    &consts,
                    internal_dim,
                    glwe_poly_size,
                    glwe_dim,
                );
            }
        }
    }

    state
}
