use super::symbolic_variance::{SymbolicVariance, VarianceOrigin};
use crate::dag::operator::{
    dot_kind, DotKind, LevelledComplexity, OperatorIndex, Precision, Shape,
};
use crate::dag::unparametrized;
use crate::noise_estimator::error;
use crate::optimization::config::NoiseBoundConfig;
use crate::utils::square;
use std::collections::HashSet;

// private short convention
use DotKind as DK;
use VarianceOrigin as VO;
type Op = unparametrized::UnparameterizedOperator;

fn first<'a, Property>(inputs: &[OperatorIndex], properties: &'a [Property]) -> &'a Property {
    &properties[inputs[0].i]
}

fn assert_all_same<Property: PartialEq + std::fmt::Debug>(
    inputs: &[OperatorIndex],
    properties: &[Property],
) {
    let first = first(inputs, properties);
    for input in inputs.iter().skip(1) {
        assert_eq!(first, &properties[input.i]);
    }
}

fn assert_inputs_uniform_precisions(
    op: &unparametrized::UnparameterizedOperator,
    out_precisions: &[Precision],
) {
    if let Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } = op {
        assert_all_same(inputs, out_precisions);
    }
}

fn assert_dot_uniform_inputs_shape(
    op: &unparametrized::UnparameterizedOperator,
    out_shapes: &[Shape],
) {
    if let Op::Dot { inputs, .. } = op {
        assert_all_same(inputs, out_shapes);
    }
}

fn assert_non_empty_inputs(op: &unparametrized::UnparameterizedOperator) {
    if let Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } = op {
        assert!(!inputs.is_empty());
    }
}

fn assert_dag_correctness(dag: &unparametrized::OperationDag) {
    for op in &dag.operators {
        assert_non_empty_inputs(op);
    }
}

fn assert_valid_variances(dag: &OperationDag) {
    for &out_variance in &dag.out_variances {
        assert!(
            SymbolicVariance::ZERO == out_variance // Special case of multiply by 0
            || 1.0 <= out_variance.input_coeff
            || 1.0 <= out_variance.lut_coeff
        );
    }
}

fn assert_properties_correctness(dag: &OperationDag) {
    for op in &dag.operators {
        assert_inputs_uniform_precisions(op, &dag.out_precisions);
        assert_dot_uniform_inputs_shape(op, &dag.out_shapes);
    }
    assert_valid_variances(dag);
}

fn variance_origin(inputs: &[OperatorIndex], out_variances: &[SymbolicVariance]) -> VarianceOrigin {
    let first_origin = first(inputs, out_variances).origin();
    for input in inputs.iter().skip(1) {
        let item = &out_variances[input.i];
        if first_origin != item.origin() {
            return VO::Mixed;
        }
    }
    first_origin
}

#[derive(Clone, Debug)]
pub struct OperationDag {
    pub operators: Vec<Op>,
    // Collect all operators ouput shape
    pub out_shapes: Vec<Shape>,
    // Collect all operators ouput precision
    pub out_precisions: Vec<Precision>,
    // Collect all operators ouput variances
    pub out_variances: Vec<SymbolicVariance>,
    pub nb_luts: u64,
    // The full dag levelled complexity
    pub levelled_complexity: LevelledComplexity,
    // Dominating variances and bounds per precision
    pub constraints_by_precisions: Vec<VariancesAndBound>,
}

#[derive(Clone, Debug)]
pub struct VariancesAndBound {
    pub precision: Precision,
    pub safe_variance_bound: f64,
    pub nb_luts: u64,
    // All final variance factor not entering a lut (usually final levelledOp)
    pub pareto_output: Vec<SymbolicVariance>,
    // All variance factor entering a lut
    pub pareto_in_lut: Vec<SymbolicVariance>,
}

impl OperationDag {
    pub fn peek_p_error(
        &self,
        input_noise_out: f64,
        blind_rotate_noise_out: f64,
        noise_keyswitch: f64,
        noise_modulus_switching: f64,
        kappa: f64,
    ) -> (f64, f64) {
        peak_p_error(
            self,
            input_noise_out,
            blind_rotate_noise_out,
            noise_keyswitch,
            noise_modulus_switching,
            kappa,
        )
    }

    pub fn feasible(
        &self,
        input_noise_out: f64,
        blind_rotate_noise_out: f64,
        noise_keyswitch: f64,
        noise_modulus_switching: f64,
    ) -> bool {
        feasible(
            self,
            input_noise_out,
            blind_rotate_noise_out,
            noise_keyswitch,
            noise_modulus_switching,
        )
    }

    pub fn complexity_cost(&self, input_lwe_dimension: u64, one_lut_cost: f64) -> f64 {
        let luts_cost = one_lut_cost * (self.nb_luts as f64);
        let levelled_cost = self.levelled_complexity.cost(input_lwe_dimension);
        luts_cost + levelled_cost
    }
}

fn out_shape(op: &unparametrized::UnparameterizedOperator, out_shapes: &mut [Shape]) -> Shape {
    match op {
        Op::Input { out_shape, .. } | Op::LevelledOp { out_shape, .. } => out_shape.clone(),
        Op::Lut { input, .. } => out_shapes[input.i].clone(),
        Op::Dot {
            inputs, weights, ..
        } => {
            if inputs.is_empty() {
                return Shape::number();
            }
            let input_shape = first(inputs, out_shapes);
            let kind = dot_kind(inputs.len() as u64, input_shape, weights);
            match kind {
                DK::Simple | DK::Tensor => Shape::number(),
                DK::CompatibleTensor => weights.shape.clone(),
                DK::Broadcast { .. } => Shape::vector(input_shape.first_dim_size()),
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
    }
}

fn out_shapes(dag: &unparametrized::OperationDag) -> Vec<Shape> {
    let nb_ops = dag.operators.len();
    let mut out_shapes = Vec::<Shape>::with_capacity(nb_ops);
    for op in &dag.operators {
        let shape = out_shape(op, &mut out_shapes);
        out_shapes.push(shape);
    }
    out_shapes
}

fn out_precision(
    op: &unparametrized::UnparameterizedOperator,
    out_precisions: &[Precision],
) -> Precision {
    match op {
        Op::Input { out_precision, .. } | Op::Lut { out_precision, .. } => *out_precision,
        Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } => out_precisions[inputs[0].i],
    }
}

fn out_precisions(dag: &unparametrized::OperationDag) -> Vec<Precision> {
    let nb_ops = dag.operators.len();
    let mut out_precisions = Vec::<Precision>::with_capacity(nb_ops);
    for op in &dag.operators {
        let precision = out_precision(op, &out_precisions);
        out_precisions.push(precision);
    }
    out_precisions
}

fn out_variance(
    op: &unparametrized::UnparameterizedOperator,
    out_shapes: &[Shape],
    out_variances: &mut [SymbolicVariance],
) -> SymbolicVariance {
    // Maintain a linear combination of input_variance and lut_out_variance
    // TODO: track each elements instead of container
    match op {
        Op::Input { .. } => SymbolicVariance::INPUT,
        Op::Lut { .. } => SymbolicVariance::LUT,
        Op::LevelledOp { inputs, manp, .. } => {
            let variance_factor = SymbolicVariance::manp_to_variance_factor(*manp);
            let origin = match variance_origin(inputs, out_variances) {
                    VO::Input => SymbolicVariance::INPUT,
                    VO::Lut | VO::Mixed /* Mixed: assume the worst */
                    => SymbolicVariance::LUT
            };
            origin * variance_factor
        }
        Op::Dot {
            inputs, weights, ..
        } => {
            let input_shape = first(inputs, out_shapes);
            let kind = dot_kind(inputs.len() as u64, input_shape, weights);
            match kind {
                DK::Simple | DK::Tensor => {
                    let first_input = inputs[0];
                    let mut out_variance = SymbolicVariance::ZERO;
                    for (j, &weight) in weights.values.iter().enumerate() {
                        let k = if kind == DK::Simple {
                            inputs[j].i
                        } else {
                            first_input.i
                        };
                        out_variance += out_variances[k] * square(weight);
                    }
                    out_variance
                }
                DK::CompatibleTensor { .. } | DK::Broadcast { .. } => todo!("TODO"),
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
    }
}

fn out_variances(
    dag: &unparametrized::OperationDag,
    out_shapes: &[Shape],
) -> Vec<SymbolicVariance> {
    let nb_ops = dag.operators.len();
    let mut out_variances = Vec::with_capacity(nb_ops);
    for op in &dag.operators {
        let vf = out_variance(op, out_shapes, &mut out_variances);
        out_variances.push(vf);
    }
    out_variances
}

fn extra_final_values_to_check(dag: &unparametrized::OperationDag) -> Vec<bool> {
    let nb_ops = dag.operators.len();
    let mut extra_values_to_check = vec![true; nb_ops];
    for op in &dag.operators {
        match op {
            Op::Input { .. } => (),
            Op::Lut { input, .. } => {
                extra_values_to_check[input.i] = false;
            }
            Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } => {
                for input in inputs {
                    extra_values_to_check[input.i] = false;
                }
            }
        }
    }
    extra_values_to_check
}

fn extra_final_variances(
    dag: &unparametrized::OperationDag,
    out_precisions: &[Precision],
    out_variances: &[SymbolicVariance],
) -> Vec<(Precision, SymbolicVariance)> {
    extra_final_values_to_check(dag)
        .iter()
        .enumerate()
        .filter_map(|(i, &is_final)| {
            if is_final {
                Some((out_precisions[i], out_variances[i]))
            } else {
                None
            }
        })
        .collect()
}

fn in_luts_variance(
    dag: &unparametrized::OperationDag,
    out_precisions: &[Precision],
    out_variances: &[SymbolicVariance],
) -> Vec<(Precision, SymbolicVariance)> {
    let only_luts = |op| {
        if let &Op::Lut { input, .. } = op {
            Some((out_precisions[input.i], out_variances[input.i]))
        } else {
            None
        }
    };
    dag.operators.iter().filter_map(only_luts).collect()
}

fn op_levelled_complexity(
    op: &unparametrized::UnparameterizedOperator,
    out_shapes: &[Shape],
) -> LevelledComplexity {
    match op {
        Op::Dot {
            inputs, weights, ..
        } => {
            let input_shape = first(inputs, out_shapes);
            let kind = dot_kind(inputs.len() as u64, input_shape, weights);
            match kind {
                DK::Simple | DK::Tensor => LevelledComplexity::ADDITION * weights.flat_size(),
                DK::CompatibleTensor { .. } | DK::Broadcast { .. } => todo!("TODO"),
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
        Op::LevelledOp { complexity, .. } => *complexity,
        Op::Input { .. } | Op::Lut { .. } => LevelledComplexity::ZERO,
    }
}

fn levelled_complexity(
    dag: &unparametrized::OperationDag,
    out_shapes: &[Shape],
) -> LevelledComplexity {
    let mut levelled_complexity = LevelledComplexity::ZERO;
    for op in &dag.operators {
        levelled_complexity += op_levelled_complexity(op, out_shapes);
    }
    levelled_complexity
}

fn safe_noise_bound(precision: Precision, noise_config: &NoiseBoundConfig) -> f64 {
    error::safe_variance_bound(
        precision as u64,
        noise_config.ciphertext_modulus_log,
        noise_config.maximum_acceptable_error_probability,
    )
}

fn constraints_by_precisions(
    out_precisions: &[Precision],
    final_variances: &[(Precision, SymbolicVariance)],
    in_luts_variance: &[(Precision, SymbolicVariance)],
    noise_config: &NoiseBoundConfig,
) -> Vec<VariancesAndBound> {
    let precisions: HashSet<Precision> = out_precisions.iter().copied().collect();
    let mut precisions: Vec<Precision> = precisions.iter().copied().collect();
    let to_noise_summary = |precision: &Precision| {
        constraint_for_one_precision(
            *precision as Precision,
            final_variances,
            in_luts_variance,
            safe_noise_bound(*precision as Precision, noise_config),
        )
    };
    // High precision first
    precisions.sort_unstable();
    precisions.iter().rev().map(to_noise_summary).collect()
}

fn select_precision<T: Copy>(target_precision: Precision, v: &[(Precision, T)]) -> Vec<T> {
    v.iter()
        .filter_map(|(p, t)| {
            if *p == target_precision {
                Some(*t)
            } else {
                None
            }
        })
        .collect()
}

fn constraint_for_one_precision(
    target_precision: Precision,
    extra_final_variances: &[(Precision, SymbolicVariance)],
    in_luts_variance: &[(Precision, SymbolicVariance)],
    safe_noise_bound: f64,
) -> VariancesAndBound {
    let extra_final_variances = select_precision(target_precision, extra_final_variances);
    let in_luts_variance = select_precision(target_precision, in_luts_variance);
    let nb_luts = in_luts_variance.len() as u64;
    let pareto_vfs_final = SymbolicVariance::reduce_to_pareto_front(extra_final_variances);
    let pareto_vfs_in_lut = SymbolicVariance::reduce_to_pareto_front(in_luts_variance);
    VariancesAndBound {
        precision: target_precision,
        safe_variance_bound: safe_noise_bound,
        nb_luts,
        pareto_output: pareto_vfs_final,
        pareto_in_lut: pareto_vfs_in_lut,
    }
}

pub fn analyze(
    dag: &unparametrized::OperationDag,
    noise_config: &NoiseBoundConfig,
) -> OperationDag {
    assert_dag_correctness(dag);
    let out_shapes = out_shapes(dag);
    let out_precisions = out_precisions(dag);
    let out_variances = out_variances(dag, &out_shapes);
    let in_luts_variance = in_luts_variance(dag, &out_precisions, &out_variances);
    let nb_luts = in_luts_variance.len() as u64;
    let extra_final_variances = extra_final_variances(dag, &out_precisions, &out_variances);
    let levelled_complexity = levelled_complexity(dag, &out_shapes);
    let constraints_by_precisions = constraints_by_precisions(
        &out_precisions,
        &extra_final_variances,
        &in_luts_variance,
        noise_config,
    );
    let result = OperationDag {
        operators: dag.operators.clone(),
        out_shapes,
        out_precisions,
        out_variances,
        nb_luts,
        levelled_complexity,
        constraints_by_precisions,
    };
    assert_properties_correctness(&result);
    result
}

fn max_update(current: &mut f64, candidate: f64) {
    if candidate > *current {
        *current = candidate;
    }
}

// Compute the maximum attained variance for the full dag
// TODO take a noise summary => peek_error or global error
fn peak_variance_per_constraint(
    constraint: &VariancesAndBound,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
) -> f64 {
    assert!(input_noise_out < blind_rotate_noise_out || blind_rotate_noise_out == 0.0);
    // the maximal variance encountered as an output that can be decrypted
    let mut variance_output = 0.0;
    for vf in &constraint.pareto_output {
        max_update(
            &mut variance_output,
            vf.eval(input_noise_out, blind_rotate_noise_out),
        );
    }
    if constraint.pareto_in_lut.is_empty() {
        return variance_output;
    }
    // the maximal variance encountered during a lut computation
    let mut variance_in_lut = 0.0;
    for vf in &constraint.pareto_in_lut {
        max_update(
            &mut variance_in_lut,
            vf.eval(input_noise_out, blind_rotate_noise_out),
        );
    }
    let peek_in_lut = variance_in_lut + noise_keyswitch + noise_modulus_switching;
    peek_in_lut.max(variance_output)
}

// Compute the maximum attained relative variance for the full dag
fn peak_relative_variance(
    dag: &OperationDag,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
) -> (f64, f64) {
    assert!(!dag.constraints_by_precisions.is_empty());
    assert!(input_noise_out <= blind_rotate_noise_out);
    let mut max_relative_var = 0.0;
    let mut safe_noise = 0.0;
    for ns in &dag.constraints_by_precisions {
        let variance_max = peak_variance_per_constraint(
            ns,
            input_noise_out,
            blind_rotate_noise_out,
            noise_keyswitch,
            noise_modulus_switching,
        );
        let relative_var = variance_max / ns.safe_variance_bound;
        if max_relative_var < relative_var {
            max_relative_var = relative_var;
            safe_noise = ns.safe_variance_bound;
        }
    }
    (max_relative_var, safe_noise)
}

fn peak_p_error(
    dag: &OperationDag,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
    kappa: f64,
) -> (f64, f64) {
    let (relative_var, variance_bound) = peak_relative_variance(
        dag,
        input_noise_out,
        blind_rotate_noise_out,
        noise_keyswitch,
        noise_modulus_switching,
    );
    let sigma_scale = kappa / relative_var.sqrt();
    (
        error::error_probability_of_sigma_scale(sigma_scale),
        relative_var * variance_bound,
    )
}

fn feasible(
    dag: &OperationDag,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
) -> bool {
    for ns in &dag.constraints_by_precisions {
        if peak_variance_per_constraint(
            ns,
            input_noise_out,
            blind_rotate_noise_out,
            noise_keyswitch,
            noise_modulus_switching,
        ) > ns.safe_variance_bound
        {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::dag::operator::{FunctionTable, LevelledComplexity, Shape, Weights};
    use crate::dag::unparametrized;
    use crate::utils::square;

    fn assert_f64_eq(v: f64, expected: f64) {
        approx::assert_relative_eq!(v, expected, epsilon = f64::EPSILON);
    }

    impl OperationDag {
        pub fn constraint(&self) -> VariancesAndBound {
            assert!(!self.constraints_by_precisions.is_empty());
            assert_eq!(self.constraints_by_precisions.len(), 1);
            self.constraints_by_precisions[0].clone()
        }
    }

    const _4_SIGMA: f64 = 1.0 - 0.999_936_657_516;

    const CONFIG: NoiseBoundConfig = NoiseBoundConfig {
        security_level: 128,
        ciphertext_modulus_log: 64,
        maximum_acceptable_error_probability: _4_SIGMA,
    };

    fn analyze(dag: &unparametrized::OperationDag) -> super::OperationDag {
        super::analyze(dag, &CONFIG)
    }

    #[test]
    fn test_1_input() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        assert_eq!(analysis.out_variances[input1.i], SymbolicVariance::INPUT);
        assert_eq!(analysis.out_shapes[input1.i], Shape::number());
        assert_eq!(analysis.levelled_complexity, LevelledComplexity::ZERO);
        assert_eq!(analysis.out_precisions[input1.i], 1);
        assert_f64_eq(complexity_cost, 0.0);
        assert!(analysis.nb_luts == 0);
        let constraint = analysis.constraint();
        assert!(constraint.pareto_output.len() == 1);
        assert_f64_eq(constraint.pareto_output[0].input_coeff, 1.0);
        assert_f64_eq(constraint.pareto_output[0].lut_coeff, 0.0);
        assert!(constraint.pareto_in_lut.is_empty());
    }

    #[test]
    fn test_1_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(8, Shape::number());
        let lut1 = graph.add_lut(input1, FunctionTable::UNKWOWN, 8);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        assert!(analysis.out_variances[lut1.i] == SymbolicVariance::LUT);
        assert!(analysis.out_shapes[lut1.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ZERO);
        assert_eq!(analysis.out_precisions[lut1.i], 8);
        assert_f64_eq(one_lut_cost, complexity_cost);
        let constraint = analysis.constraint();
        assert!(constraint.pareto_output.len() == 1);
        assert!(constraint.pareto_in_lut.len() == 1);
        assert_f64_eq(constraint.pareto_output[0].input_coeff, 0.0);
        assert_f64_eq(constraint.pareto_output[0].lut_coeff, 1.0);
        assert_f64_eq(constraint.pareto_in_lut[0].input_coeff, 1.0);
        assert_f64_eq(constraint.pareto_in_lut[0].lut_coeff, 0.0);
    }

    #[test]
    fn test_1_dot() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let weights = Weights::vector([1, 2]);
        let norm2: f64 = 1.0 * 1.0 + 2.0 * 2.0;
        let dot = graph.add_dot([input1, input1], weights);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        let expected_var = SymbolicVariance {
            input_coeff: norm2,
            lut_coeff: 0.0,
        };
        assert!(analysis.out_variances[dot.i] == expected_var);
        assert!(analysis.out_shapes[dot.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION * 2);
        assert_eq!(analysis.out_precisions[dot.i], 1);
        let expected_dot_cost = (2 * lwe_dim) as f64;
        assert_f64_eq(expected_dot_cost, complexity_cost);
        let constraint = analysis.constraint();
        assert!(constraint.pareto_in_lut.is_empty());
        assert!(constraint.pareto_output.len() == 1);
        assert_f64_eq(constraint.pareto_output[0].input_coeff, 5.0);
        assert_f64_eq(constraint.pareto_output[0].lut_coeff, 0.0);
    }

    #[test]
    fn test_1_dot_levelled() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(3, Shape::number());
        let cpx_dot = LevelledComplexity::ADDITION;
        let weights = Weights::vector([1, 2]);
        let manp = 1.0 * 1.0 + 2.0 * 2_f64;
        let dot = graph.add_levelled_op([input1, input1], cpx_dot, manp, Shape::number(), "dot");
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        assert!(analysis.out_variances[dot.i].origin() == VO::Input);
        assert_eq!(analysis.out_precisions[dot.i], 3);
        let expected_square_norm2 = weights.square_norm2() as f64;
        let actual_square_norm2 = analysis.out_variances[dot.i].input_coeff;
        // Due to call on log2() to compute manp the result is not exact
        assert_f64_eq(actual_square_norm2, expected_square_norm2);
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION);
        assert_f64_eq(lwe_dim as f64, complexity_cost);
        let constraint = analysis.constraint();
        assert!(constraint.pareto_in_lut.is_empty());
        assert!(constraint.pareto_output.len() == 1);
        assert_eq!(constraint.pareto_output[0].origin(), VO::Input);
        assert_f64_eq(constraint.pareto_output[0].input_coeff, 5.0);
    }

    #[test]
    fn test_dot_tensorized_lut_dot_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::vector(2));
        let weights = &Weights::vector([1, 2]);
        let dot1 = graph.add_dot([input1], weights);
        let lut1 = graph.add_lut(dot1, FunctionTable::UNKWOWN, 1);
        let dot2 = graph.add_dot([lut1, lut1], weights);
        let lut2 = graph.add_lut(dot2, FunctionTable::UNKWOWN, 1);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        let expected_var_dot1 = SymbolicVariance {
            input_coeff: weights.square_norm2() as f64,
            lut_coeff: 0.0,
        };
        let expected_var_lut1 = SymbolicVariance {
            input_coeff: 0.0,
            lut_coeff: 1.0,
        };
        let expected_var_dot2 = SymbolicVariance {
            input_coeff: 0.0,
            lut_coeff: weights.square_norm2() as f64,
        };
        let expected_var_lut2 = SymbolicVariance {
            input_coeff: 0.0,
            lut_coeff: 1.0,
        };
        assert!(analysis.out_variances[dot1.i] == expected_var_dot1);
        assert!(analysis.out_variances[lut1.i] == expected_var_lut1);
        assert!(analysis.out_variances[dot2.i] == expected_var_dot2);
        assert!(analysis.out_variances[lut2.i] == expected_var_lut2);
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION * 4);
        let expected_cost = (lwe_dim * 4) as f64 + 2.0 * one_lut_cost;
        assert_f64_eq(expected_cost, complexity_cost);
        let constraint = analysis.constraint();
        assert_eq!(constraint.pareto_output.len(), 1);
        assert_eq!(constraint.pareto_output[0].origin(), VO::Lut);
        assert_f64_eq(constraint.pareto_output[0].lut_coeff, 1.0);
        assert_eq!(constraint.pareto_in_lut.len(), 1);
        assert_eq!(constraint.pareto_in_lut[0].origin(), VO::Lut);
        assert_f64_eq(
            constraint.pareto_in_lut[0].lut_coeff,
            weights.square_norm2() as f64,
        );
    }

    #[test]
    fn test_lut_dot_mixed_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let lut1 = graph.add_lut(input1, FunctionTable::UNKWOWN, 1);
        let weights = &Weights::vector([2, 3]);
        let dot1 = graph.add_dot([input1, lut1], weights);
        let _lut2 = graph.add_lut(dot1, FunctionTable::UNKWOWN, 1);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        let expected_cost = (2 * lwe_dim) as f64 + 2.0 * one_lut_cost;
        assert_f64_eq(expected_cost, complexity_cost);
        let expected_mixed = SymbolicVariance {
            input_coeff: square(weights.values[0] as f64),
            lut_coeff: square(weights.values[1] as f64),
        };
        let constraint = analysis.constraint();
        assert_eq!(constraint.pareto_output.len(), 1);
        assert_eq!(constraint.pareto_output[0], SymbolicVariance::LUT);
        assert_eq!(constraint.pareto_in_lut.len(), 1);
        assert_eq!(constraint.pareto_in_lut[0].origin(), VO::Mixed);
        assert_eq!(constraint.pareto_in_lut[0], expected_mixed);
    }

    #[test]
    fn test_multi_precision_input() {
        let mut graph = unparametrized::OperationDag::new();
        let max_precision: Precision = 5;
        for i in 1..=max_precision {
            let _ = graph.add_input(i as u8, Shape::number());
        }
        let analysis = analyze(&graph);
        assert!(analysis.constraints_by_precisions.len() == max_precision as usize);
        let mut prev_safe_noise_bound = 0.0;
        for (i, ns) in analysis.constraints_by_precisions.iter().enumerate() {
            let i_prec = i as Precision;
            assert_eq!(ns.precision, max_precision - i_prec);
            assert_f64_eq(ns.pareto_output[0].input_coeff, 1.0);
            assert!(prev_safe_noise_bound < ns.safe_variance_bound);
            prev_safe_noise_bound = ns.safe_variance_bound;
        }
    }

    #[test]
    fn test_multi_precision_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let max_precision: Precision = 5;
        for p in 1..=max_precision {
            let input = graph.add_input(p, Shape::number());
            let _lut = graph.add_lut(input, FunctionTable::UNKWOWN, p);
        }
        let analysis = analyze(&graph);
        assert!(analysis.constraints_by_precisions.len() == max_precision as usize);
        let mut prev_safe_noise_bound = 0.0;
        for (i, ns) in analysis.constraints_by_precisions.iter().enumerate() {
            let i_prec = i as Precision;
            assert_eq!(ns.precision, max_precision - i_prec);
            assert_eq!(ns.pareto_output.len(), 1);
            assert_eq!(ns.pareto_in_lut.len(), 1);
            assert_f64_eq(ns.pareto_output[0].input_coeff, 0.0);
            assert_f64_eq(ns.pareto_output[0].lut_coeff, 1.0);
            assert_f64_eq(ns.pareto_in_lut[0].input_coeff, 1.0);
            assert_f64_eq(ns.pareto_in_lut[0].lut_coeff, 0.0);
            assert!(prev_safe_noise_bound < ns.safe_variance_bound);
            prev_safe_noise_bound = ns.safe_variance_bound;
        }
    }
}
