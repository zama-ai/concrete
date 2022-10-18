use super::analyze;
use crate::dag::operator::{LevelledComplexity, Precision};
use crate::dag::unparametrized;
use crate::noise_estimator::error;
use crate::noise_estimator::operators::atomic_pattern as noise_atomic_pattern;
use crate::optimization::atomic_pattern::{
    Caches, OptimizationDecompositionsConsts, OptimizationState, Solution,
};
use crate::optimization::config::{Config, NoiseBoundConfig, SearchSpace};
use crate::optimization::decomposition::PersistDecompCache;
use crate::parameters::GlweParameters;
use crate::security;

use concrete_commons::dispersion::DispersionParameter;

#[allow(clippy::too_many_lines)]
fn update_best_solution_with_best_decompositions(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    dag: &analyze::OperationDag,
    internal_dim: u64,
    glwe_params: GlweParameters,
    input_noise_out: f64,
    noise_modulus_switching: f64,
    caches: &mut Caches,
) {
    assert!(dag.nb_luts > 0);
    let glwe_poly_size = glwe_params.polynomial_size();
    let input_lwe_dimension = glwe_params.glwe_dimension * glwe_poly_size;

    let mut best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let mut best_variance = state.best_solution.map_or(f64::INFINITY, |s| s.noise_max);
    let mut best_p_error = state.best_solution.map_or(f64::INFINITY, |s| s.p_error);

    let br_pareto = caches
        .blind_rotate
        .pareto_quantities(glwe_params, internal_dim);

    let ks_pareto = caches
        .keyswitch
        .pareto_quantities(glwe_params, internal_dim);

    // by constructon br_pareto and ks_pareto are non-empty
    let mut best_br = br_pareto[0];
    let mut best_ks = ks_pareto[0];
    let mut update_best_solution = false;

    for &br_quantity in br_pareto {
        // increasing complexity, decreasing variance
        let one_lut_cost = br_quantity.complexity;
        let complexity = dag.complexity(input_lwe_dimension, one_lut_cost);
        if complexity > best_complexity {
            // Since br_pareto is scanned by increasing complexity, we can stop
            break;
        }
        let not_feasible = !dag.feasible(
            input_noise_out,
            br_quantity.noise,
            0.0,
            noise_modulus_switching,
        );
        if not_feasible {
            continue;
        }
        for &ks_quantity in ks_pareto {
            let one_lut_cost = ks_quantity.complexity + br_quantity.complexity;
            let complexity = dag.complexity(input_lwe_dimension, one_lut_cost);
            let worse_complexity = complexity > best_complexity;
            if worse_complexity {
                // Since ks_pareto is scanned by increasing complexity, we can stop
                break;
            }
            let not_feasible = !dag.feasible(
                input_noise_out,
                br_quantity.noise,
                ks_quantity.noise,
                noise_modulus_switching,
            );
            if not_feasible {
                continue;
            }

            let (peek_p_error, variance) = dag.peek_p_error(
                input_noise_out,
                br_quantity.noise,
                ks_quantity.noise,
                noise_modulus_switching,
                consts.kappa,
            );
            #[allow(clippy::float_cmp)]
            let same_comlexity_no_few_errors =
                complexity == best_complexity && peek_p_error >= best_p_error;
            if same_comlexity_no_few_errors {
                continue;
            }

            // The complexity is either better or equivalent with less errors
            update_best_solution = true;
            best_complexity = complexity;
            best_p_error = peek_p_error;
            best_variance = variance;
            best_br = br_quantity;
            best_ks = ks_quantity;
        }
    } // br ks

    if update_best_solution {
        state.best_solution = Some(Solution {
            input_lwe_dimension,
            internal_ks_output_lwe_dimension: internal_dim,
            ks_decomposition_level_count: best_ks.decomp.level,
            ks_decomposition_base_log: best_ks.decomp.log2_base,
            glwe_polynomial_size: glwe_params.polynomial_size(),
            glwe_dimension: glwe_params.glwe_dimension,
            br_decomposition_level_count: best_br.decomp.level,
            br_decomposition_base_log: best_br.decomp.log2_base,
            complexity: best_complexity,
            p_error: best_p_error,
            global_p_error: dag.global_p_error(
                input_noise_out,
                best_br.noise,
                best_ks.noise,
                noise_modulus_switching,
                consts.kappa,
            ),
            noise_max: best_variance,
        });
    }
}

const REL_EPSILON_PROBA: f64 = 1.0 + 1e-8;

fn update_no_luts_solution(
    state: &mut OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    dag: &analyze::OperationDag,
    glwe_params: GlweParameters,
    input_noise_out: f64,
) {
    const CHECKED_IGNORED_NOISE: f64 = f64::MAX;
    const UNDEFINED_PARAM: u64 = 0;

    let input_lwe_dimension = glwe_params.sample_extract_lwe_dimension();

    let best_complexity = state.best_solution.map_or(f64::INFINITY, |s| s.complexity);
    let best_p_error = state.best_solution.map_or(f64::INFINITY, |s| s.p_error);

    let complexity = if dag.levelled_complexity == LevelledComplexity::ZERO {
        // The compiler has given a 0 levelled complexity.
        // There is no way to compare solutions.
        // Assuming linear complexity.
        input_lwe_dimension as f64
    } else {
        dag.levelled_complexity(input_lwe_dimension)
    };

    if complexity > best_complexity {
        return;
    }

    let (p_error, variance) = dag.peek_p_error(
        input_noise_out,
        CHECKED_IGNORED_NOISE,
        CHECKED_IGNORED_NOISE,
        CHECKED_IGNORED_NOISE,
        consts.kappa,
    );

    #[allow(clippy::float_cmp)]
    let same_complexity_no_few_errors = complexity == best_complexity && p_error >= best_p_error;
    if same_complexity_no_few_errors {
        return;
    }
    // The complexity is either better or equivalent with less errors
    state.best_solution = Some(Solution {
        input_lwe_dimension,
        internal_ks_output_lwe_dimension: UNDEFINED_PARAM,
        ks_decomposition_level_count: UNDEFINED_PARAM,
        ks_decomposition_base_log: UNDEFINED_PARAM,
        glwe_polynomial_size: glwe_params.polynomial_size(),
        glwe_dimension: glwe_params.glwe_dimension,
        br_decomposition_level_count: UNDEFINED_PARAM,
        br_decomposition_base_log: UNDEFINED_PARAM,
        complexity,
        p_error,
        global_p_error: dag.global_p_error(
            input_noise_out,
            CHECKED_IGNORED_NOISE,
            CHECKED_IGNORED_NOISE,
            CHECKED_IGNORED_NOISE,
            consts.kappa,
        ),
        noise_max: variance,
    });
}

fn minimal_variance(config: &Config, glwe_params: GlweParameters) -> f64 {
    security::glwe::minimal_variance(
        glwe_params,
        config.ciphertext_modulus_log,
        config.security_level,
    )
    .get_variance()
}

fn optimize_no_luts(
    mut state: OptimizationState,
    consts: &OptimizationDecompositionsConsts,
    dag: &analyze::OperationDag,
    search_space: &SearchSpace,
) -> OptimizationState {
    let not_feasible = |input_noise_out| !dag.feasible(input_noise_out, 0.0, 0.0, 0.0);
    for &glwe_dim in &search_space.glwe_dimensions {
        for &glwe_log_poly_size in &search_space.glwe_log_polynomial_sizes {
            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };
            let input_noise_out = minimal_variance(&consts.config, glwe_params);
            if not_feasible(input_noise_out) {
                continue;
            }
            update_no_luts_solution(&mut state, consts, dag, glwe_params, input_noise_out);
        }
    }
    state
}

#[allow(clippy::too_many_lines)]
pub fn optimize(
    dag: &unparametrized::OperationDag,
    config: Config,
    search_space: &SearchSpace,
    cache: &PersistDecompCache,
) -> OptimizationState {
    let ciphertext_modulus_log = config.ciphertext_modulus_log;
    let security_level = config.security_level;
    let noise_config = NoiseBoundConfig {
        security_level,
        maximum_acceptable_error_probability: config.maximum_acceptable_error_probability,
        ciphertext_modulus_log,
    };
    let &min_precision = dag.out_precisions.iter().min().unwrap();

    let dag = analyze::analyze(dag, &noise_config);

    let safe_variance = error::safe_variance_bound_2padbits(
        min_precision as u64,
        ciphertext_modulus_log,
        config.maximum_acceptable_error_probability,
    );
    let kappa =
        error::sigma_scale_of_error_probability(config.maximum_acceptable_error_probability);

    let consts = OptimizationDecompositionsConsts {
        config,
        kappa,
        sum_size: 0,            // superseeded by dag.complexity_cost
        noise_factor: f64::NAN, // superseeded by dag.lut_variance_max
        safe_variance,
    };

    let mut state = OptimizationState {
        best_solution: None,
    };

    if dag.nb_luts == 0 {
        return optimize_no_luts(state, &consts, &dag, search_space);
    }
    let mut caches = Caches::new(cache);

    let noise_modulus_switching = |glwe_poly_size, internal_lwe_dimensions| {
        noise_atomic_pattern::estimate_modulus_switching_noise_with_binary_key(
            internal_lwe_dimensions,
            glwe_poly_size,
            ciphertext_modulus_log,
        )
        .get_variance()
    };

    let not_feasible = |input_noise_out, noise_modulus_switching| {
        !dag.feasible(input_noise_out, 0.0, 0.0, noise_modulus_switching)
    };

    for &glwe_dim in &search_space.glwe_dimensions {
        for &glwe_log_poly_size in &search_space.glwe_log_polynomial_sizes {
            let glwe_params = GlweParameters {
                log2_polynomial_size: glwe_log_poly_size,
                glwe_dimension: glwe_dim,
            };
            let input_noise_out = minimal_variance(&config, glwe_params);
            for &internal_dim in &search_space.internal_lwe_dimensions {
                let glwe_poly_size = 1 << glwe_log_poly_size;
                let noise_modulus_switching = noise_modulus_switching(glwe_poly_size, internal_dim);
                if not_feasible(input_noise_out, noise_modulus_switching) {
                    // noise_modulus_switching is increasing with internal_dim
                    break;
                }
                update_best_solution_with_best_decompositions(
                    &mut state,
                    &consts,
                    &dag,
                    internal_dim,
                    glwe_params,
                    input_noise_out,
                    noise_modulus_switching,
                    &mut caches,
                );
                if dag.nb_luts == 0 && state.best_solution.is_some() {
                    return state;
                }
            }
        }
    }

    caches.backport_to(cache);

    if let Some(sol) = state.best_solution {
        assert!(0.0 <= sol.p_error && sol.p_error <= 1.0);
        assert!(0.0 <= sol.global_p_error && sol.global_p_error <= 1.0);
        assert!(sol.p_error <= config.maximum_acceptable_error_probability * REL_EPSILON_PROBA);
        assert!(sol.p_error <= sol.global_p_error * REL_EPSILON_PROBA);
    }

    state
}

pub fn optimize_v0(
    sum_size: u64,
    precision: u64,
    config: Config,
    noise_factor: f64,
    search_space: &SearchSpace,
    cache: &PersistDecompCache,
) -> OptimizationState {
    use crate::dag::operator::{FunctionTable, Shape};
    let same_scale_manp = 0.0;
    let manp = noise_factor;
    let out_shape = &Shape::number();
    let complexity = LevelledComplexity::ADDITION * sum_size;
    let comment = "dot";
    let mut dag = unparametrized::OperationDag::new();
    let precision = precision as Precision;
    let input1 = dag.add_input(precision, out_shape);
    let dot1 = dag.add_levelled_op([input1], complexity, same_scale_manp, out_shape, comment);
    let lut1 = dag.add_lut(dot1, FunctionTable::UNKWOWN, precision);
    let dot2 = dag.add_levelled_op([lut1], complexity, manp, out_shape, comment);
    let _lut2 = dag.add_lut(dot2, FunctionTable::UNKWOWN, precision);
    let mut state = optimize(&dag, config, search_space, cache);
    if let Some(sol) = &mut state.best_solution {
        sol.complexity /= 2.0;
    }
    state
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use once_cell::sync::Lazy;

    use super::*;
    use crate::computing_cost::cpu::CpuComplexity;
    use crate::config;
    use crate::dag::operator::{FunctionTable, Shape, Weights};
    use crate::noise_estimator::p_error::repeat_p_error;
    use crate::optimization::config::SearchSpace;
    use crate::optimization::dag::solo_key::symbolic_variance::VarianceOrigin;
    use crate::optimization::{atomic_pattern, decomposition};
    use crate::utils::square;

    fn small_relative_diff(v1: f64, v2: f64) -> bool {
        f64::abs(v1 - v2) / f64::max(v1, v2) <= 0.000_000_1
    }

    impl Solution {
        fn assert_same_pbs_solution(&self, other: Self) -> bool {
            let mut other = other;
            other.global_p_error = self.global_p_error;
            if small_relative_diff(self.noise_max, other.noise_max)
                && small_relative_diff(self.p_error, other.p_error)
            {
                other.noise_max = self.noise_max;
                other.p_error = self.p_error;
            }
            assert_eq!(self, &other);
            self == &other
        }
    }

    const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;

    static SHARED_CACHES: Lazy<PersistDecompCache> = Lazy::new(|| {
        let processing_unit = config::ProcessingUnit::Cpu;
        decomposition::cache(128, processing_unit, None)
    });

    fn optimize(dag: &unparametrized::OperationDag) -> OptimizationState {
        let config = Config {
            security_level: 128,
            maximum_acceptable_error_probability: _4_SIGMA,
            ciphertext_modulus_log: 64,
            complexity_model: &CpuComplexity::default(),
        };

        let search_space = SearchSpace::default_cpu();

        super::optimize(dag, config, &search_space, &SHARED_CACHES)
    }

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
        let processing_unit = config::ProcessingUnit::Cpu;

        let search_space = SearchSpace::default(processing_unit);

        let sum_size = 1;

        let config = Config {
            security_level: 128,
            maximum_acceptable_error_probability: _4_SIGMA,
            ciphertext_modulus_log: 64,
            complexity_model: &CpuComplexity::default(),
        };

        let _ = optimize_v0(
            sum_size,
            precision,
            config,
            weight as f64,
            &search_space,
            &SHARED_CACHES,
        );
        // ensure cache is filled

        let chrono = Instant::now();
        let state = optimize_v0(
            sum_size,
            precision,
            config,
            weight as f64,
            &search_space,
            &SHARED_CACHES,
        );

        times.dag_time += chrono.elapsed().as_nanos();
        let chrono = Instant::now();
        let state_ref = atomic_pattern::optimize_one(
            sum_size,
            precision,
            config,
            weight as f64,
            &search_space,
            &SHARED_CACHES,
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
        assert!(sol.assert_same_pbs_solution(sol_ref));
        assert!(!sol.global_p_error.is_nan());
        assert!(sol.p_error <= sol.global_p_error);
        assert!(sol.global_p_error <= 1.0);
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

    fn v0_parameter_ref_with_dot(precision: Precision, weight: i64) {
        let processing_unit = config::ProcessingUnit::Cpu;
        let security_level = 128;

        let mut dag = unparametrized::OperationDag::new();
        {
            let input1 = dag.add_input(precision, Shape::number());
            let dot1 = dag.add_dot([input1], [1]);
            let lut1 = dag.add_lut(dot1, FunctionTable::UNKWOWN, precision);
            let dot2 = dag.add_dot([lut1], [weight]);
            let _lut2 = dag.add_lut(dot2, FunctionTable::UNKWOWN, precision);
        }
        {
            let dag2 = analyze::analyze(
                &dag,
                &NoiseBoundConfig {
                    security_level,
                    maximum_acceptable_error_probability: _4_SIGMA,
                    ciphertext_modulus_log: 64,
                },
            );
            let constraint = dag2.constraint();
            assert_eq!(constraint.pareto_output.len(), 1);
            assert_eq!(constraint.pareto_in_lut.len(), 1);
            assert_eq!(constraint.pareto_output[0].origin(), VarianceOrigin::Lut);
            assert_f64_eq(1.0, constraint.pareto_output[0].lut_coeff);
            assert!(constraint.pareto_in_lut.len() == 1);
            assert_eq!(constraint.pareto_in_lut[0].origin(), VarianceOrigin::Lut);
            assert_f64_eq(square(weight) as f64, constraint.pareto_in_lut[0].lut_coeff);
        }

        let search_space = SearchSpace::default(processing_unit);

        let config = Config {
            security_level,
            maximum_acceptable_error_probability: _4_SIGMA,
            ciphertext_modulus_log: 64,
            complexity_model: &CpuComplexity::default(),
        };

        let state = optimize(&dag);
        let state_ref = atomic_pattern::optimize_one(
            1,
            precision as u64,
            config,
            weight as f64,
            &search_space,
            &SHARED_CACHES,
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
        assert!(sol.assert_same_pbs_solution(sol_ref));
        assert!(!sol.global_p_error.is_nan());
        assert!(sol.p_error <= sol.global_p_error);
        assert!(sol.global_p_error <= 1.0);
    }

    fn no_lut_vs_lut(precision: Precision) {
        let mut dag_lut = unparametrized::OperationDag::new();
        let input1 = dag_lut.add_input(precision, Shape::number());
        let _lut1 = dag_lut.add_lut(input1, FunctionTable::UNKWOWN, precision);

        let mut dag_no_lut = unparametrized::OperationDag::new();
        let _input2 = dag_no_lut.add_input(precision, Shape::number());

        let state_no_lut = optimize(&dag_no_lut);
        let state_lut = optimize(&dag_lut);
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

    fn lut_with_input_base_noise_better_than_lut_with_lut_base_noise(
        precision: Precision,
        weight: i64,
    ) {
        let weight = &Weights::number(weight);

        let mut dag_1 = unparametrized::OperationDag::new();
        {
            let input1 = dag_1.add_input(precision, Shape::number());
            let scaled_input1 = dag_1.add_dot([input1], weight);
            let lut1 = dag_1.add_lut(scaled_input1, FunctionTable::UNKWOWN, precision);
            let _lut2 = dag_1.add_lut(lut1, FunctionTable::UNKWOWN, precision);
        }

        let mut dag_2 = unparametrized::OperationDag::new();
        {
            let input1 = dag_2.add_input(precision, Shape::number());
            let lut1 = dag_2.add_lut(input1, FunctionTable::UNKWOWN, precision);
            let scaled_lut1 = dag_2.add_dot([lut1], weight);
            let _lut2 = dag_2.add_lut(scaled_lut1, FunctionTable::UNKWOWN, precision);
        }

        let state_1 = optimize(&dag_1);
        let state_2 = optimize(&dag_2);

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

    fn lut_1_layer_has_better_complexity(precision: Precision) {
        let dag_1_layer = {
            let mut dag = unparametrized::OperationDag::new();
            let input1 = dag.add_input(precision, Shape::number());
            let _lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, precision);
            let _lut2 = dag.add_lut(input1, FunctionTable::UNKWOWN, precision);
            dag
        };
        let dag_2_layer = {
            let mut dag = unparametrized::OperationDag::new();
            let input1 = dag.add_input(precision, Shape::number());
            let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, precision);
            let _lut2 = dag.add_lut(lut1, FunctionTable::UNKWOWN, precision);
            dag
        };

        let sol_1_layer = optimize(&dag_1_layer).best_solution.unwrap();
        let sol_2_layer = optimize(&dag_2_layer).best_solution.unwrap();
        assert!(
            sol_1_layer.complexity <= sol_2_layer.complexity,
            "Precision: {} => sol_1_layer: {} ; sol_2_layer: {}",
            precision,
            sol_1_layer.complexity,
            sol_2_layer.complexity
        );
    }

    #[test]
    fn test_lut_1_layer_is_better() {
        // for some reason on 4, 5, 6, the complexity is already minimal
        // this could be due to pre-defined pareto set
        for precision in [1, 2, 3, 7, 8] {
            lut_1_layer_has_better_complexity(precision);
        }
    }

    fn circuit(dag: &mut unparametrized::OperationDag, precision: Precision, weight: i64) {
        let input = dag.add_input(precision, Shape::number());
        let dot1 = dag.add_dot([input], [weight]);
        let lut1 = dag.add_lut(dot1, FunctionTable::UNKWOWN, precision);
        let dot2 = dag.add_dot([lut1], [weight]);
        let _lut2 = dag.add_lut(dot2, FunctionTable::UNKWOWN, precision);
    }

    fn assert_multi_precision_dominate_single(weight: i64) -> Option<bool> {
        let low_precision = 4u8;
        let high_precision = 5u8;
        let mut dag_low = unparametrized::OperationDag::new();
        let mut dag_high = unparametrized::OperationDag::new();
        let mut dag_multi = unparametrized::OperationDag::new();

        {
            circuit(&mut dag_low, low_precision, weight);
            circuit(&mut dag_high, high_precision, 1);
            circuit(&mut dag_multi, low_precision, weight);
            circuit(&mut dag_multi, high_precision, 1);
        }
        let state_multi = optimize(&dag_multi);

        let mut sol_multi = state_multi.best_solution?;

        let state_low = optimize(&dag_low);
        let state_high = optimize(&dag_high);

        let sol_low = state_low.best_solution.unwrap();
        let sol_high = state_high.best_solution.unwrap();
        sol_multi.complexity /= 2.0;
        if sol_low.complexity < sol_high.complexity {
            assert!(sol_high.assert_same_pbs_solution(sol_multi));
            Some(true)
        } else {
            assert!(
                sol_low.complexity < sol_multi.complexity
                    || sol_low.assert_same_pbs_solution(sol_multi)
            );
            Some(false)
        }
    }

    #[test]
    fn test_multi_precision_dominate_single() {
        let mut prev = Some(true); // true -> ... -> true -> false -> ... -> false
        for log2_weight in 0..29 {
            let weight = 1 << log2_weight;
            let current = assert_multi_precision_dominate_single(weight);
            #[allow(clippy::match_like_matches_macro)] // less readable
            let authorized = match (prev, current) {
                (Some(false), Some(true)) => false,
                (None, Some(_)) => false,
                _ => true,
            };
            assert!(authorized);
            prev = current;
        }
    }

    fn local_to_approx_global_p_error(local_p_error: f64, nb_pbs: u64) -> f64 {
        #[allow(clippy::float_cmp)]
        if local_p_error == 1f64 {
            return 1.0;
        }
        #[allow(clippy::float_cmp)]
        if local_p_error == 0f64 {
            return 0.0;
        }

        assert!(local_p_error <= 1.0);
        assert!(0.0 <= local_p_error);

        repeat_p_error(local_p_error, nb_pbs)
    }

    #[test]
    fn test_global_p_error_input() {
        for precision in [4_u8, 8] {
            for weight in [1, 3, 27, 243, 729] {
                for dim in [1, 2, 16, 32] {
                    let _ = check_global_p_error_input(dim, weight, precision);
                }
            }
        }
    }

    fn check_global_p_error_input(dim: u64, weight: i64, precision: u8) -> f64 {
        let shape = Shape::vector(dim);
        let weights = Weights::number(weight);
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(precision, shape);
        let _dot1 = dag.add_dot([input1], weights); // this is just several multiply
        let state = optimize(&dag);
        let sol = state.best_solution.unwrap();
        let worst_expected_p_error_dim = local_to_approx_global_p_error(sol.p_error, dim);
        approx::assert_relative_eq!(sol.global_p_error, worst_expected_p_error_dim);
        sol.global_p_error
    }

    #[test]
    fn test_global_p_error_lut() {
        for precision in [4_u8, 8] {
            for weight in [1, 3, 27, 243, 729] {
                for depth in [2, 16, 32] {
                    check_global_p_error_lut(depth, weight, precision);
                }
            }
        }
    }

    fn check_global_p_error_lut(depth: u64, weight: i64, precision: u8) {
        let shape = Shape::number();
        let weights = Weights::number(weight);
        let mut dag = unparametrized::OperationDag::new();
        let mut last_val = dag.add_input(precision, shape);
        for _i in 0..depth {
            let dot = dag.add_dot([last_val], &weights);
            last_val = dag.add_lut(dot, FunctionTable::UNKWOWN, precision);
        }
        let state = optimize(&dag);
        let sol = state.best_solution.unwrap();
        // the first lut on input has reduced impact on error probability
        let lower_nb_dominating_lut = depth - 1;
        let lower_global_p_error =
            local_to_approx_global_p_error(sol.p_error, lower_nb_dominating_lut);
        let higher_global_p_error =
            local_to_approx_global_p_error(sol.p_error, lower_nb_dominating_lut + 1);
        assert!(lower_global_p_error <= sol.global_p_error);
        assert!(sol.global_p_error <= higher_global_p_error);
    }

    fn dag_2_precisions_lut_chain(
        depth: u64,
        precision_low: Precision,
        precision_high: Precision,
        weight_low: i64,
        weight_high: i64,
    ) -> unparametrized::OperationDag {
        let shape = Shape::number();
        let mut dag = unparametrized::OperationDag::new();
        let weights_low = Weights::number(weight_low);
        let weights_high = Weights::number(weight_high);
        let mut last_val_low = dag.add_input(precision_low, &shape);
        let mut last_val_high = dag.add_input(precision_high, &shape);
        for _i in 0..depth {
            let dot_low = dag.add_dot([last_val_low], &weights_low);
            last_val_low = dag.add_lut(dot_low, FunctionTable::UNKWOWN, precision_low);
            let dot_high = dag.add_dot([last_val_high], &weights_high);
            last_val_high = dag.add_lut(dot_high, FunctionTable::UNKWOWN, precision_high);
        }
        dag
    }

    #[allow(clippy::unnecessary_cast)] // clippy bug refusing as Precision on const
    #[test]
    fn test_global_p_error_dominating_lut() {
        let depth = 128;
        let weights_low = 1;
        let weights_high = 1;
        let precision_low = 6 as Precision;
        let precision_high = 8 as Precision;
        let dag = dag_2_precisions_lut_chain(
            depth,
            precision_low,
            precision_high,
            weights_low,
            weights_high,
        );
        let sol = optimize(&dag).best_solution.unwrap();
        // the 2 first luts and low precision/weight luts have little impact on error probability
        let nb_dominating_lut = depth - 1;
        let approx_global_p_error = local_to_approx_global_p_error(sol.p_error, nb_dominating_lut);
        // errors rate is approximated accurately
        approx::assert_relative_eq!(
            sol.global_p_error,
            approx_global_p_error,
            max_relative = 1e-01
        );
    }

    #[allow(clippy::unnecessary_cast)] // clippy bug refusing as Precision on const
    #[test]
    fn test_global_p_error_non_dominating_lut() {
        let depth = 128;
        let weights_low = 1024 * 2200;
        let weights_high = 1;
        let precision_low = 6 as Precision;
        let precision_high = 8 as Precision;
        let dag = dag_2_precisions_lut_chain(
            depth,
            precision_low,
            precision_high,
            weights_low,
            weights_high,
        );
        let sol = optimize(&dag).best_solution.unwrap();
        // all intern luts have an impact on error probability almost equaly
        let nb_dominating_lut = (2 * depth) - 1;
        let approx_global_p_error = local_to_approx_global_p_error(sol.p_error, nb_dominating_lut);
        // errors rate is approximated accurately
        approx::assert_relative_eq!(
            sol.global_p_error,
            approx_global_p_error,
            max_relative = 0.05
        );
    }

    fn circuit_with_rounded_lut(
        rounded_precision: Precision,
        precision: Precision,
        weight: i64,
    ) -> unparametrized::OperationDag {
        // circuit with intermediate high precision in levelled op
        let shape = Shape::number();
        let mut dag = unparametrized::OperationDag::new();
        let weight = Weights::number(weight);
        let val = dag.add_input(precision, &shape);
        let lut1 = dag.add_rounded_lut(val, FunctionTable::UNKWOWN, rounded_precision, precision);
        let dot = dag.add_dot([lut1], &weight);
        let _lut2 = dag.add_rounded_lut(
            dot,
            FunctionTable::UNKWOWN,
            rounded_precision,
            rounded_precision,
        );
        dag
    }

    fn check_global_p_error_rounded_lut(
        precision: Precision,
        rounded_precision: Precision,
        weight: i64,
        check_linear_speedup: bool,
    ) {
        let dag_no_rounded = circuit_with_rounded_lut(precision, precision, weight);
        let dag_rounded = circuit_with_rounded_lut(rounded_precision, precision, weight);
        let dag_reduced = circuit_with_rounded_lut(rounded_precision, rounded_precision, weight);
        let best_reduced = optimize(&dag_reduced).best_solution.unwrap();
        let best_rounded = optimize(&dag_rounded).best_solution.unwrap();
        let best_no_rounded_complexity = optimize(&dag_no_rounded)
            .best_solution
            .map_or(f64::INFINITY, |s| s.complexity);
        // println!("Slowdown acc {rounded_precision} -> {precision}, {best_rounded.complexity/best_reduced.complexity}");
        // println!("Speedup tlu {precision} -> {rounded_precision}, {best_no_rounded_complexity/best_rounded.complexity}");
        if weight == 0 && precision - rounded_precision <= 4
            || weight == 16 && precision - rounded_precision <= 3
        {
            // linear slowdown with almost no margin
            assert!(
                best_rounded.complexity
                    <= best_reduced.complexity
                        * (1.0 + 1.01 * (precision - rounded_precision) as f64)
            );
        } else if precision - rounded_precision <= 4 {
            // linear slowdown with margin
            assert!(
                best_rounded.complexity
                    <= best_reduced.complexity
                        * (1.0 + 1.5 * (precision - rounded_precision) as f64)
            );
        } else if precision != rounded_precision {
            // slowdown
            assert!(best_reduced.complexity < best_rounded.complexity);
        }
        // linear speedup
        if check_linear_speedup {
            assert!(
                best_rounded.complexity * (precision - rounded_precision) as f64
                    <= best_no_rounded_complexity
            );
        }
    }

    #[allow(clippy::unnecessary_cast)] // clippy bug refusing as Precision on const
    #[test]
    fn test_global_p_error_rounded_lut() {
        let precision = 8 as Precision;
        for rounded_precision in 4..9 {
            check_global_p_error_rounded_lut(precision, rounded_precision, 1, true);
        }
    }

    #[allow(clippy::unnecessary_cast)] // clippy bug refusing as Precision on const
    #[test]
    fn test_global_p_error_increased_accumulator() {
        let rounded_precision = 8 as Precision;
        for precision in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] {
            for weight in [1, 2, 4, 8, 16, 32, 64, 128] {
                println!("{precision} {weight}");
                check_global_p_error_rounded_lut(precision, rounded_precision, weight, false);
            }
        }
    }
}
