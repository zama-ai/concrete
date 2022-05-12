use concrete_commons::dispersion::{DispersionParameter, Variance};
use concrete_commons::numeric::UnsignedInteger;

use crate::dag::operator::LevelledComplexity;
use crate::dag::unparametrized;
use crate::noise_estimator::error;
use crate::noise_estimator::error::error_probability_of_sigma_scale;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;

use crate::optimization::atomic_pattern::{
    pareto_blind_rotate, pareto_keyswitch, OptimizationDecompositionsConsts, OptimizationState,
    Solution,
};

use crate::parameters::{BrDecompositionParameters, GlweParameters, KsDecompositionParameters};
use crate::pareto;
use crate::security::glwe::minimal_variance;
use crate::utils::square;

use super::analyze;

const CUTS: bool = true;
const PARETO_CUTS: bool = true;
const CROSS_PARETO_CUTS: bool = PARETO_CUTS && true;

#[allow(clippy::too_many_lines)]
fn update_best_solution_with_best_decompositions<W: UnsignedInteger>(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    dag: &analyze::OperationDag,
    internal_dim: u64,
    glwe_params: GlweParameters,
    noise_modulus_switching: f64,
) {
    let safe_variance = consts.safe_variance;
    let glwe_poly_size = glwe_params.polynomial_size();
    let input_lwe_dimension = glwe_params.glwe_dimension * glwe_poly_size;

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let mut best_lut_complexity = state
        .best_solution
        .map_or(f64::INFINITY, |s| s.lut_complexity);
    let mut best_variance = state.best_solution.map_or(f64::INFINITY, |s| s.noise_max);

    let mut cut_complexity =
        (best_complexity - dag.complexity_cost(input_lwe_dimension, 0.0)) / (dag.nb_luts as f64);
    let mut cut_noise = safe_variance - noise_modulus_switching;

    if dag.nb_luts == 0 {
        cut_noise = f64::INFINITY;
        cut_complexity = f64::INFINITY;
    }

    let br_pareto =
        pareto_blind_rotate::<W>(consts, internal_dim, glwe_params, cut_complexity, cut_noise);
    if br_pareto.is_empty() {
        return;
    }
    if PARETO_CUTS {
        cut_noise -= br_pareto[br_pareto.len() - 1].noise;
        cut_complexity -= br_pareto[0].complexity;
    }

    let ks_pareto = pareto_keyswitch::<W>(
        consts,
        input_lwe_dimension,
        internal_dim,
        cut_complexity,
        cut_noise,
    );
    if ks_pareto.is_empty() {
        return;
    }

    let i_max_ks = ks_pareto.len() - 1;
    let mut i_current_max_ks = i_max_ks;
    let input_noise_out = minimal_variance(
        glwe_params,
        consts.ciphertext_modulus_log,
        consts.security_level,
    )
    .get_variance();

    let mut best_br_i = 0;
    let mut best_ks_i = 0;
    let mut update_best_solution = false;

    for br_quantity in br_pareto {
        // increasing complexity, decreasing variance
        let peek_variance = dag.peek_variance(
            input_noise_out,
            br_quantity.noise,
            0.0,
            noise_modulus_switching,
        );
        if peek_variance > safe_variance && CUTS {
            continue;
        }
        let one_pbs_cost = br_quantity.complexity;
        let complexity = dag.complexity_cost(input_lwe_dimension, one_pbs_cost);
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
            let ks_quantity = ks_pareto[i_ks_pareto];
            let peek_variance = dag.peek_variance(
                input_noise_out,
                br_quantity.noise,
                ks_quantity.noise,
                noise_modulus_switching,
            );
            // let noise_max = br_quantity.noise * dag.lut_base_noise_worst_lut + ks_quantity.noise + noise_modulus_switching;
            if peek_variance > safe_variance {
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
            }
            let one_lut_cost = ks_quantity.complexity + br_quantity.complexity;
            let complexity = dag.complexity_cost(input_lwe_dimension, one_lut_cost);

            let better_complexity = complexity < best_complexity;
            #[allow(clippy::float_cmp)]
            let same_complexity_with_less_errors =
                complexity == best_complexity && peek_variance < best_variance;
            if better_complexity || same_complexity_with_less_errors {
                best_lut_complexity = one_lut_cost;
                best_complexity = complexity;
                best_variance = peek_variance;
                best_br_i = br_quantity.index;
                best_ks_i = ks_quantity.index;
                update_best_solution = true;
            }
        }
    } // br ks

    if update_best_solution {
        let sigma = Variance(safe_variance).get_standard_dev() * consts.kappa;
        let sigma_scale = sigma / Variance(best_variance).get_standard_dev();
        let p_error = error_probability_of_sigma_scale(sigma_scale);
        let BrDecompositionParameters {
            level: br_l,
            log2_base: br_b,
        } = consts.blind_rotate_decompositions[best_br_i];
        let KsDecompositionParameters {
            level: ks_l,
            log2_base: ks_b,
        } = consts.keyswitch_decompositions[best_ks_i];
        state.best_solution = Some(Solution {
            input_lwe_dimension,
            internal_ks_output_lwe_dimension: internal_dim,
            ks_decomposition_level_count: ks_l,
            ks_decomposition_base_log: ks_b,
            glwe_polynomial_size: glwe_params.polynomial_size(),
            glwe_dimension: glwe_params.glwe_dimension,
            br_decomposition_level_count: br_l,
            br_decomposition_base_log: br_b,
            noise_max: best_variance,
            complexity: best_complexity,
            lut_complexity: best_lut_complexity,
            p_error,
        });
    }
}

const REL_EPSILON_PROBA: f64 = 1.0 + 1e-8;

#[allow(clippy::too_many_lines)]
pub fn optimize<W: UnsignedInteger>(
    dag: &unparametrized::OperationDag,
    security_level: u64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
) -> OptimizationState {
    let ciphertext_modulus_log = W::BITS as u64;
    let dag = analyze::analyze(dag);

    let &max_precision = dag.out_precisions.iter().max().unwrap();

    let safe_variance = error::variance_max(
        max_precision as u64,
        ciphertext_modulus_log,
        maximum_acceptable_error_probability,
    );
    let kappa = error::sigma_scale_of_error_probability(maximum_acceptable_error_probability);

    let consts = OptimizationDecompositionsConsts {
        kappa,
        sum_size: 0, // superseeded by dag.complexity_cost
        security_level,
        noise_factor: f64::NAN, // superseeded by dag.lut_variance_max
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

    let noise_modulus_switching = |glwe_poly_size, internal_lwe_dimensions| {
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key::<W>(
            internal_lwe_dimensions,
            glwe_poly_size,
        )
        .get_variance()
    };

    for &glwe_dim in glwe_dimensions {
        for &glwe_log_poly_size in glwe_log_polynomial_sizes {
            let glwe_poly_size = 1 << glwe_log_poly_size;
            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };
            for &internal_dim in internal_lwe_dimensions {
                let noise_modulus_switching = noise_modulus_switching(glwe_poly_size, internal_dim);
                if CUTS && noise_modulus_switching > consts.safe_variance {
                    // assume this noise is increasing with internal_dim
                    break;
                }
                update_best_solution_with_best_decompositions::<W>(
                    &mut state,
                    &consts,
                    &dag,
                    internal_dim,
                    glwe_params,
                    noise_modulus_switching,
                );
                if dag.nb_luts == 0 && state.best_solution.is_some() {
                    return state;
                }
            }
        }
    }

    if let Some(sol) = state.best_solution {
        assert!(0.0 <= sol.p_error && sol.p_error <= 1.0);
        assert!(sol.p_error <= maximum_acceptable_error_probability * REL_EPSILON_PROBA);
    }

    state
}

pub fn optimize_v0<W: UnsignedInteger>(
    sum_size: u64,
    precision: u64,
    security_level: u64,
    noise_factor: f64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
) -> OptimizationState {
    use crate::dag::operator::{FunctionTable, Shape};
    let same_scale_manp = 0.0;
    let manp = square(noise_factor);
    let out_shape = &Shape::number();
    let complexity = LevelledComplexity::ADDITION * sum_size;
    let comment = "dot";
    let mut dag = unparametrized::OperationDag::new();
    let input1 = dag.add_input(precision as u8, out_shape);
    let dot1 = dag.add_levelled_op([input1], complexity, same_scale_manp, out_shape, comment);
    let lut1 = dag.add_lut(dot1, FunctionTable::UNKWOWN);
    let dot2 = dag.add_levelled_op([lut1], complexity, manp, out_shape, comment);
    let _lut2 = dag.add_lut(dot2, FunctionTable::UNKWOWN);
    let mut state = optimize::<u64>(
        &dag,
        security_level,
        maximum_acceptable_error_probability,
        glwe_log_polynomial_sizes,
        glwe_dimensions,
        internal_lwe_dimensions,
    );
    if let Some(sol) = &mut state.best_solution {
        sol.complexity /= 2.0;
    }
    state
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::dag::operator::{FunctionTable, Shape, Weights};
    use crate::global_parameters::DEFAUT_DOMAINS;
    use crate::optimization::dag::solo_key::symbolic_variance::VarianceOrigin;
    use crate::utils::square;

    use super::*;
    use crate::optimization::atomic_pattern;

    fn small_relative_diff(v1: f64, v2: f64) -> bool {
        f64::abs(v1 - v2) / f64::max(v1, v2) <= f64::EPSILON
    }

    impl Solution {
        fn same(&self, other: Self) -> bool {
            let mut other = other;
            if small_relative_diff(self.noise_max, other.noise_max)
                && small_relative_diff(self.p_error, other.p_error)
            {
                other.noise_max = self.noise_max;
                other.p_error = self.p_error;
            }
            self == &other
        }
    }

    const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;

    struct Times {
        worst_time: u128,
        dag_time: u128,
    }

    fn assert_f64_eq(v: f64, expected: f64) {
        approx::assert_relative_eq!(v, expected, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_v0_parameter_ref() {
        let mut times = Times {
            worst_time: 0,
            dag_time: 0,
        };
        for log_weight in 0..=16 {
            let weight = 1 << log_weight;
            for precision in 1..=9 {
                v0_parameter_ref(precision, weight, &mut times);
            }
        }
        assert!(times.worst_time * 2 > times.dag_time);
    }

    fn v0_parameter_ref(precision: u64, weight: u64, times: &mut Times) {
        let security_level = 128;
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let sum_size = 1;
        let maximum_acceptable_error_probability = _4_SIGMA;

        let chrono = Instant::now();
        let state = optimize_v0::<u64>(
            sum_size,
            precision,
            security_level,
            weight as f64,
            maximum_acceptable_error_probability,
            &glwe_log_polynomial_sizes,
            &glwe_dimensions,
            &internal_lwe_dimensions,
        );
        times.dag_time += chrono.elapsed().as_nanos();
        let chrono = Instant::now();
        let state_ref = atomic_pattern::optimize_one::<u64>(
            sum_size,
            precision,
            security_level,
            weight as f64,
            maximum_acceptable_error_probability,
            &glwe_log_polynomial_sizes,
            &glwe_dimensions,
            &internal_lwe_dimensions,
            None,
        );
        times.worst_time += chrono.elapsed().as_nanos();
        assert_eq!(
            state.best_solution.is_some(),
            state_ref.best_solution.is_some()
        );
        if state.best_solution.is_none() {
            return;
        }
        let sol = state.best_solution.unwrap();
        let sol_ref = state_ref.best_solution.unwrap();
        assert!(sol.same(sol_ref));
    }

    #[test]
    fn test_v0_parameter_ref_with_dot() {
        for log_weight in 0..=16 {
            let weight = 1 << log_weight;
            for precision in 1..=9 {
                v0_parameter_ref_with_dot(precision, weight);
            }
        }
    }

    fn v0_parameter_ref_with_dot(precision: u64, weight: u64) {
        let mut dag = unparametrized::OperationDag::new();
        {
            let input1 = dag.add_input(precision as u8, Shape::number());
            let dot1 = dag.add_dot([input1], [1]);
            let lut1 = dag.add_lut(dot1, FunctionTable::UNKWOWN);
            let dot2 = dag.add_dot([lut1], [weight]);
            let _lut2 = dag.add_lut(dot2, FunctionTable::UNKWOWN);
        }
        {
            let dag2 = analyze::analyze(&dag);
            let summary = dag2.noise_summary;
            assert_eq!(summary.pareto_vfs_final.len(), 1);
            assert_eq!(summary.pareto_vfs_in_lut.len(), 1);
            assert_eq!(summary.pareto_vfs_final[0].origin(), VarianceOrigin::Lut);
            assert_f64_eq(1.0, summary.pareto_vfs_final[0].lut_vf);
            assert!(summary.pareto_vfs_in_lut.len() == 1);
            assert_eq!(summary.pareto_vfs_in_lut[0].origin(), VarianceOrigin::Lut);
            assert_f64_eq(square(weight) as f64, summary.pareto_vfs_in_lut[0].lut_vf);
        }

        let security_level = 128;
        let maximum_acceptable_error_probability = _4_SIGMA;
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let state = optimize::<u64>(
            &dag,
            security_level,
            maximum_acceptable_error_probability,
            &glwe_log_polynomial_sizes,
            &glwe_dimensions,
            &internal_lwe_dimensions,
        );
        let state_ref = atomic_pattern::optimize_one::<u64>(
            1,
            precision,
            security_level,
            weight as f64,
            maximum_acceptable_error_probability,
            &glwe_log_polynomial_sizes,
            &glwe_dimensions,
            &internal_lwe_dimensions,
            None,
        );
        assert_eq!(
            state.best_solution.is_some(),
            state_ref.best_solution.is_some()
        );
        if state.best_solution.is_none() {
            return;
        }
        let sol = state.best_solution.unwrap();
        let mut sol_ref = state_ref.best_solution.unwrap();
        sol_ref.complexity *= 2.0 /* number of luts */;
        assert!(sol.same(sol_ref));
    }

    fn no_lut_vs_lut(precision: u64) {
        let mut dag_lut = unparametrized::OperationDag::new();
        let input1 = dag_lut.add_input(precision as u8, Shape::number());
        let _lut1 = dag_lut.add_lut(input1, FunctionTable::UNKWOWN);

        let mut dag_no_lut = unparametrized::OperationDag::new();
        let _input2 = dag_no_lut.add_input(precision as u8, Shape::number());

        let security_level = 128;
        let maximum_acceptable_error_probability = _4_SIGMA;
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();

        let opt = |dag: &unparametrized::OperationDag| {
            optimize::<u64>(
                dag,
                security_level,
                maximum_acceptable_error_probability,
                &glwe_log_polynomial_sizes,
                &glwe_dimensions,
                &internal_lwe_dimensions,
            )
        };

        let state_no_lut = opt(&dag_no_lut);
        let state_lut = opt(&dag_lut);
        assert_eq!(
            state_no_lut.best_solution.is_some(),
            state_lut.best_solution.is_some()
        );

        if state_lut.best_solution.is_none() {
            return;
        }

        let sol_no_lut = state_no_lut.best_solution.unwrap();
        let sol_lut = state_lut.best_solution.unwrap();
        assert!(sol_no_lut.complexity < sol_lut.complexity);
    }
    #[test]
    fn test_lut_vs_no_lut() {
        for precision in 1..=8 {
            no_lut_vs_lut(precision);
        }
    }

    fn lut_with_input_base_noise_better_than_lut_with_lut_base_noise(precision: u64, weight: u64) {
        let weight = &Weights::number(weight);

        let mut dag_1 = unparametrized::OperationDag::new();
        {
            let input1 = dag_1.add_input(precision as u8, Shape::number());
            let scaled_input1 = dag_1.add_dot([input1], weight);
            let lut1 = dag_1.add_lut(scaled_input1, FunctionTable::UNKWOWN);
            let _lut2 = dag_1.add_lut(lut1, FunctionTable::UNKWOWN);
        }

        let mut dag_2 = unparametrized::OperationDag::new();
        {
            let input1 = dag_2.add_input(precision as u8, Shape::number());
            let lut1 = dag_2.add_lut(input1, FunctionTable::UNKWOWN);
            let scaled_lut1 = dag_2.add_dot([lut1], weight);
            let _lut2 = dag_2.add_lut(scaled_lut1, FunctionTable::UNKWOWN);
        }

        let security_level = 128;
        let maximum_acceptable_error_probability = _4_SIGMA;
        let glwe_log_polynomial_sizes: Vec<u64> = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions: Vec<u64> = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();

        let opt = |dag: &unparametrized::OperationDag| {
            optimize::<u64>(
                dag,
                security_level,
                maximum_acceptable_error_probability,
                &glwe_log_polynomial_sizes,
                &glwe_dimensions,
                &internal_lwe_dimensions,
            )
        };

        let state_1 = opt(&dag_1);
        let state_2 = opt(&dag_2);

        if state_1.best_solution.is_none() {
            assert!(state_2.best_solution.is_none());
            return;
        }
        let sol_1 = state_1.best_solution.unwrap();
        let sol_2 = state_2.best_solution.unwrap();
        assert!(sol_1.complexity < sol_2.complexity || sol_1.p_error < sol_2.p_error);
    }

    #[test]
    fn test_lut_with_input_base_noise_better_than_lut_with_lut_base_noise() {
        for log_weight in 1..=16 {
            let weight = 1 << log_weight;
            for precision in 5..=9 {
                lut_with_input_base_noise_better_than_lut_with_lut_base_noise(precision, weight);
            }
        }
    }
}
