#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::style)]
#![allow(clippy::cast_precision_loss)] // u64 to f64
#![allow(clippy::cast_possible_truncation)] // u64 to usize
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::cast_lossless)]
#![warn(unused_results)]

pub mod computing_cost;

pub mod global_parameters;
pub mod graph;
pub mod noise_estimator;
pub mod optimisation;
pub mod parameters;
pub mod pareto;
pub mod security;
pub mod weight;

#[no_mangle]
pub extern "C" fn optimise_bootstrap(
    precision: u64,
    security_level: u64,
    noise_factor: f64,
    maximum_acceptable_error_probability: f64,
) -> optimisation::atomic_pattern::Solution {
    use global_parameters::DEFAUT_DOMAINS;
    let sum_size = 1;
    let glwe_log_polynomial_sizes = DEFAUT_DOMAINS
        .glwe_pbs_constrained
        .log2_polynomial_size
        .as_vec();
    let glwe_dimensions = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
    let internal_lwe_dimensions = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
    let result = optimisation::atomic_pattern::optimise_one::<u64>(
        sum_size,
        precision,
        security_level,
        noise_factor,
        maximum_acceptable_error_probability,
        &glwe_log_polynomial_sizes,
        &glwe_dimensions,
        &internal_lwe_dimensions,
        None,
    );
    match result.best_solution {
        Some(solution) => solution,
        None => optimisation::atomic_pattern::Solution {
            input_lwe_dimension: 0,
            internal_ks_output_lwe_dimension: 0,
            ks_decomposition_level_count: 0,
            ks_decomposition_base_log: 0,
            glwe_polynomial_size: 0,
            glwe_dimension: 0,
            br_decomposition_level_count: 0,
            br_decomposition_base_log: 0,
            complexity: 0.0,
            noise_max: 0.0,
            p_error: 1.0, // error probability
        },
    }
}
