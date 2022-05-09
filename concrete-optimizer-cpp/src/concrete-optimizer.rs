fn no_solution() -> ffi::Solution {
    ffi::Solution {
        p_error: 1.0, // error probability to signal an impossible solution
        ..ffi::Solution::default()
    }
}

fn optimise_bootstrap(
    precision: u64,
    security_level: u64,
    noise_factor: f64,
    maximum_acceptable_error_probability: f64,
) -> ffi::Solution {
    use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
    let sum_size = 1;
    let glwe_log_polynomial_sizes = DEFAUT_DOMAINS
        .glwe_pbs_constrained
        .log2_polynomial_size
        .as_vec();
    let glwe_dimensions = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
    let internal_lwe_dimensions = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
    let result = concrete_optimizer::optimisation::atomic_pattern::optimise_one::<u64>(
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
    result.best_solution.map_or_else(no_solution, |a| a.into())
}

impl From<concrete_optimizer::optimisation::atomic_pattern::Solution> for ffi::Solution {
    fn from(a: concrete_optimizer::optimisation::atomic_pattern::Solution) -> Self {
        Self {
            input_lwe_dimension: a.input_lwe_dimension,
            internal_ks_output_lwe_dimension: a.internal_ks_output_lwe_dimension,
            ks_decomposition_level_count: a.ks_decomposition_level_count,
            ks_decomposition_base_log: a.ks_decomposition_base_log,
            glwe_polynomial_size: a.glwe_polynomial_size,
            glwe_dimension: a.glwe_dimension,
            br_decomposition_level_count: a.br_decomposition_level_count,
            br_decomposition_base_log: a.br_decomposition_base_log,
            complexity: a.complexity,
            noise_max: a.noise_max,
            p_error: a.p_error,
        }
    }
}

#[cxx::bridge]
mod ffi {
    #[namespace = "concrete_optimizer"]
    extern "Rust" {
        fn optimise_bootstrap(
            precision: u64,
            security_level: u64,
            noise_factor: f64,
            maximum_acceptable_error_probability: f64,
        ) -> Solution;
    }

    #[namespace = "concrete_optimizer"]
    #[derive(Debug, Clone, Copy, Default)]
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
}
