use concrete_optimizer::graph::operator::{
    self, FunctionTable, LevelledComplexity, OperatorIndex, Shape,
};
use concrete_optimizer::graph::unparametrized;

fn no_solution() -> ffi::Solution {
    ffi::Solution {
        p_error: 1.0, // error probability to signal an impossible solution
        ..ffi::Solution::default()
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

pub struct OperationDag(unparametrized::OperationDag);

fn empty() -> Box<OperationDag> {
    Box::new(OperationDag(unparametrized::OperationDag::new()))
}

impl OperationDag {
    fn add_input(&mut self, out_precision: u8, out_shape: &[u64]) -> ffi::OperatorIndex {
        let out_shape = Shape {
            dimensions_size: out_shape.to_owned(),
        };

        self.0.add_input(out_precision, out_shape).into()
    }

    fn add_lut(&mut self, input: ffi::OperatorIndex, table: &[u64]) -> ffi::OperatorIndex {
        let table = FunctionTable {
            values: table.to_owned(),
        };

        self.0.add_lut(input.into(), table).into()
    }

    #[allow(clippy::boxed_local)]
    fn add_dot(
        &mut self,
        inputs: &[ffi::OperatorIndex],
        weights: Box<Weights>,
    ) -> ffi::OperatorIndex {
        let inputs: Vec<OperatorIndex> = inputs.iter().copied().map(Into::into).collect();

        self.0.add_dot(&inputs, &weights.0).into()
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
            .add_levelled_op(&inputs, complexity, manp, out_shape, comment)
            .into()
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

        type OperationDag;

        #[namespace = "concrete_optimizer::dag"]
        fn empty() -> Box<OperationDag>;

        fn add_input(
            self: &mut OperationDag,
            out_precision: u8,
            out_shape: &[u64],
        ) -> OperatorIndex;

        fn add_lut(self: &mut OperationDag, input: OperatorIndex, table: &[u64]) -> OperatorIndex;

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
}
