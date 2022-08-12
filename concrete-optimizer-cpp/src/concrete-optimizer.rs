use concrete_optimizer::dag::operator::{
    self, FunctionTable, LevelledComplexity, OperatorIndex, Precision, Shape,
};
use concrete_optimizer::dag::unparametrized;

use concrete_optimizer::optimization::dag::solo_key::optimize_generic::Solution as DagSolution;

fn no_solution() -> ffi::Solution {
    ffi::Solution {
        p_error: 1.0, // error probability to signal an impossible solution
        ..ffi::Solution::default()
    }
}

fn no_dag_solution() -> ffi::DagSolution {
    ffi::DagSolution {
        p_error: 1.0, // error probability to signal an impossible solution
        ..ffi::DagSolution::default()
    }
}

fn optimize_bootstrap(
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
    let result = concrete_optimizer::optimization::atomic_pattern::optimize_one::<u64>(
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
    result
        .best_solution
        .map_or_else(no_solution, |solution| solution.into())
}

fn convert_to_dag_solution(sol: &ffi::Solution) -> ffi::DagSolution {
    sol.into()
}

impl From<&ffi::Solution> for ffi::DagSolution {
    fn from(sol: &ffi::Solution) -> Self {
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
            global_p_error: f64::NAN,
            use_wop_pbs: false,
            cb_decomposition_level_count: 0,
            cb_decomposition_base_log: 0,
            crt_decomposition: vec![],
        }
    }
}

impl From<concrete_optimizer::optimization::atomic_pattern::Solution> for ffi::Solution {
    fn from(a: concrete_optimizer::optimization::atomic_pattern::Solution) -> Self {
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

impl From<DagSolution> for ffi::DagSolution {
    fn from(sol: DagSolution) -> Self {
        match sol {
            DagSolution::WpSolution(sol) => Self {
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
                use_wop_pbs: false,
                cb_decomposition_level_count: 0,
                cb_decomposition_base_log: 0,
                crt_decomposition: vec![],
            },
            DagSolution::WopSolution(sol) => Self {
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
                global_p_error: f64::NAN,
                use_wop_pbs: true,
                cb_decomposition_level_count: sol.cb_decomposition_level_count,
                cb_decomposition_base_log: sol.cb_decomposition_base_log,
                crt_decomposition: sol.crt_decomposition,
            },
        }
    }
}

pub struct OperationDag(unparametrized::OperationDag);

fn empty() -> Box<OperationDag> {
    Box::new(OperationDag(unparametrized::OperationDag::new()))
}

impl OperationDag {
    fn add_input(&mut self, out_precision: Precision, out_shape: &[u64]) -> ffi::OperatorIndex {
        let out_shape = Shape {
            dimensions_size: out_shape.to_owned(),
        };

        self.0.add_input(out_precision, out_shape).into()
    }

    fn add_lut(
        &mut self,
        input: ffi::OperatorIndex,
        table: &[u64],
        out_precision: Precision,
    ) -> ffi::OperatorIndex {
        let table = FunctionTable {
            values: table.to_owned(),
        };

        self.0.add_lut(input.into(), table, out_precision).into()
    }

    #[allow(clippy::boxed_local)]
    fn add_dot(
        &mut self,
        inputs: &[ffi::OperatorIndex],
        weights: Box<Weights>,
    ) -> ffi::OperatorIndex {
        let inputs: Vec<OperatorIndex> = inputs.iter().copied().map(Into::into).collect();

        self.0.add_dot(inputs, weights.0).into()
    }

    fn add_levelled_op(
        &mut self,
        inputs: &[ffi::OperatorIndex],
        lwe_dim_cost_factor: f64,
        fixed_cost: f64,
        manp: f64,
        out_shape: &[u64],
        comment: &str,
    ) -> ffi::OperatorIndex {
        let inputs: Vec<OperatorIndex> = inputs.iter().copied().map(Into::into).collect();

        let out_shape = Shape {
            dimensions_size: out_shape.to_owned(),
        };

        let complexity = LevelledComplexity {
            lwe_dim_cost_factor,
            fixed_cost,
        };

        self.0
            .add_levelled_op(inputs, complexity, manp, out_shape, comment)
            .into()
    }

    fn optimize_v0(
        &self,
        security_level: u64,
        maximum_acceptable_error_probability: f64,
    ) -> ffi::Solution {
        use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
        let glwe_log_polynomial_sizes = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let result = concrete_optimizer::optimization::dag::solo_key::optimize::optimize::<u64>(
            &self.0,
            security_level,
            maximum_acceptable_error_probability,
            &glwe_log_polynomial_sizes,
            &glwe_dimensions,
            &internal_lwe_dimensions,
        );
        result
            .best_solution
            .map_or_else(no_solution, |solution| solution.into())
    }

    fn optimize(
        &self,
        security_level: u64,
        maximum_acceptable_error_probability: f64,
        default_log_norm2_woppbs: f64,
    ) -> ffi::DagSolution {
        use concrete_optimizer::global_parameters::DEFAUT_DOMAINS;
        let glwe_log_polynomial_sizes = DEFAUT_DOMAINS
            .glwe_pbs_constrained
            .log2_polynomial_size
            .as_vec();
        let glwe_dimensions = DEFAUT_DOMAINS.glwe_pbs_constrained.glwe_dimension.as_vec();
        let internal_lwe_dimensions = DEFAUT_DOMAINS.free_glwe.glwe_dimension.as_vec();
        let result =
            concrete_optimizer::optimization::dag::solo_key::optimize_generic::optimize::<u64>(
                &self.0,
                security_level,
                maximum_acceptable_error_probability,
                &glwe_log_polynomial_sizes,
                &glwe_dimensions,
                &internal_lwe_dimensions,
                default_log_norm2_woppbs,
            );
        result.map_or_else(no_dag_solution, |solution| solution.into())
    }

    fn dump(&self) -> String {
        self.0.dump()
    }
}

pub struct Weights(operator::Weights);

fn vector(weights: &[u64]) -> Box<Weights> {
    Box::new(Weights(operator::Weights::vector(weights)))
}

impl From<OperatorIndex> for ffi::OperatorIndex {
    fn from(oi: OperatorIndex) -> Self {
        Self { index: oi.i }
    }
}

#[allow(clippy::from_over_into)]
impl Into<OperatorIndex> for ffi::OperatorIndex {
    fn into(self) -> OperatorIndex {
        OperatorIndex { i: self.index }
    }
}

#[cxx::bridge]
mod ffi {

    #[namespace = "concrete_optimizer"]
    extern "Rust" {

        #[namespace = "concrete_optimizer::v0"]
        fn optimize_bootstrap(
            precision: u64,
            security_level: u64,
            noise_factor: f64,
            maximum_acceptable_error_probability: f64,
        ) -> Solution;

        #[namespace = "concrete_optimizer::utils"]
        fn convert_to_dag_solution(solution: &Solution) -> DagSolution;

        type OperationDag;

        #[namespace = "concrete_optimizer::dag"]
        fn empty() -> Box<OperationDag>;

        fn add_input(
            self: &mut OperationDag,
            out_precision: u8,
            out_shape: &[u64],
        ) -> OperatorIndex;

        fn add_lut(
            self: &mut OperationDag,
            input: OperatorIndex,
            table: &[u64],
            out_precision: u8,
        ) -> OperatorIndex;

        fn add_dot(
            self: &mut OperationDag,
            inputs: &[OperatorIndex],
            weights: Box<Weights>,
        ) -> OperatorIndex;

        fn add_levelled_op(
            self: &mut OperationDag,
            inputs: &[OperatorIndex],
            lwe_dim_cost_factor: f64,
            fixed_cost: f64,
            manp: f64,
            out_shape: &[u64],
            comment: &str,
        ) -> OperatorIndex;

        fn optimize_v0(
            self: &OperationDag,
            security_level: u64,
            maximum_acceptable_error_probability: f64,
        ) -> Solution;

        fn optimize(
            self: &OperationDag,
            security_level: u64,
            maximum_acceptable_error_probability: f64,
            default_log_norm2_woppbs: f64,
        ) -> DagSolution;

        fn dump(self: &OperationDag) -> String;

        type Weights;

        #[namespace = "concrete_optimizer::weights"]
        fn vector(weights: &[u64]) -> Box<Weights>;
    }

    #[derive(Clone, Copy)]
    #[namespace = "concrete_optimizer::dag"]
    struct OperatorIndex {
        index: usize,
    }

    #[namespace = "concrete_optimizer::v0"]
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

    #[namespace = "concrete_optimizer::dag"]
    #[derive(Debug, Clone, Default)]
    pub struct DagSolution {
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
        pub use_wop_pbs: bool,
        pub cb_decomposition_level_count: u64,
        pub cb_decomposition_base_log: u64,
        pub crt_decomposition: Vec<u64>,
    }
}
