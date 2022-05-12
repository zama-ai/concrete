use super::symbolic_variance::{SymbolicVariance, VarianceOrigin};
use crate::dag::operator::{
    dot_kind, DotKind, LevelledComplexity, OperatorIndex, Precision, Shape,
};
use crate::dag::unparametrized;
use crate::utils::square;

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
            || 1.0 <= out_variance.input_vf
            || 1.0 <= out_variance.lut_vf
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
    // Global summaries of worst noise cases
    pub noise_summary: NoiseSummary,
}

#[derive(Clone, Debug)]
pub struct NoiseSummary {
    // All final variance factor not entering a lut (usually final levelledOp)
    pub pareto_vfs_final: Vec<SymbolicVariance>,
    // All variance factor entering a lut
    pub pareto_vfs_in_lut: Vec<SymbolicVariance>,
}

impl OperationDag {
    pub fn peek_variance(
        &self,
        input_noise_out: f64,
        blind_rotate_noise_out: f64,
        noise_keyswitch: f64,
        noise_modulus_switching: f64,
    ) -> f64 {
        peek_variance(
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
    out_precisions: &mut [Precision],
) -> Precision {
    match op {
        Op::Input { out_precision, .. } => *out_precision,
        Op::Lut { input, .. } => out_precisions[input.i],
        Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } => out_precisions[inputs[0].i],
    }
}

fn out_precisions(dag: &unparametrized::OperationDag) -> Vec<Precision> {
    let nb_ops = dag.operators.len();
    let mut out_precisions = Vec::<Precision>::with_capacity(nb_ops);
    for op in &dag.operators {
        let precision = out_precision(op, &mut out_precisions);
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
    out_variances: &[SymbolicVariance],
) -> Vec<SymbolicVariance> {
    extra_final_values_to_check(dag)
        .iter()
        .enumerate()
        .filter_map(|(i, &is_final)| {
            if is_final {
                Some(out_variances[i])
            } else {
                None
            }
        })
        .collect()
}

fn in_luts_variance(
    dag: &unparametrized::OperationDag,
    out_variances: &[SymbolicVariance],
) -> Vec<SymbolicVariance> {
    let only_luts = |op| {
        if let &Op::Lut { input, .. } = op {
            Some(out_variances[input.i])
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

fn max_update(current: &mut f64, candidate: f64) {
    if candidate > *current {
        *current = candidate;
    }
}

fn noise_summary(
    final_variances: Vec<SymbolicVariance>,
    in_luts_variance: Vec<SymbolicVariance>,
) -> NoiseSummary {
    let pareto_vfs_final = SymbolicVariance::reduce_to_pareto_front(final_variances);
    let pareto_vfs_in_lut = SymbolicVariance::reduce_to_pareto_front(in_luts_variance);
    NoiseSummary {
        pareto_vfs_final,
        pareto_vfs_in_lut,
    }
}

pub fn analyze(dag: &unparametrized::OperationDag) -> OperationDag {
    assert_dag_correctness(dag);
    let out_shapes = out_shapes(dag);
    let out_precisions = out_precisions(dag);
    let out_variances = out_variances(dag, &out_shapes);
    let in_luts_variance = in_luts_variance(dag, &out_variances);
    let nb_luts = in_luts_variance.len() as u64;
    let extra_final_variances = extra_final_variances(dag, &out_variances);
    let levelled_complexity = levelled_complexity(dag, &out_shapes);
    let noise_summary = noise_summary(extra_final_variances, in_luts_variance);
    let result = OperationDag {
        operators: dag.operators.clone(),
        out_shapes,
        out_precisions,
        out_variances,
        nb_luts,
        levelled_complexity,
        noise_summary,
    };
    assert_properties_correctness(&result);
    result
}

// Compute the maximum attained variance for the full dag
// TODO take a noise summary => peek_error or global error
fn peek_variance(
    dag: &OperationDag,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
) -> f64 {
    assert!(input_noise_out < blind_rotate_noise_out);
    let mut variance_peek_final = 0.0; // updated by the loop
    for vf in &dag.noise_summary.pareto_vfs_final {
        max_update(
            &mut variance_peek_final,
            vf.eval(input_noise_out, blind_rotate_noise_out),
        );
    }

    let mut variance_peek_in_lut = 0.0; // updated by the loop
    for vf in &dag.noise_summary.pareto_vfs_in_lut {
        max_update(
            &mut variance_peek_in_lut,
            vf.eval(input_noise_out, blind_rotate_noise_out),
        );
    }
    let peek_in_lut = variance_peek_in_lut + noise_keyswitch + noise_modulus_switching;
    peek_in_lut.max(variance_peek_final)
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
        let summary = analysis.noise_summary;
        assert!(summary.pareto_vfs_final.len() == 1);
        assert_f64_eq(summary.pareto_vfs_final[0].input_vf, 1.0);
        assert_f64_eq(summary.pareto_vfs_final[0].lut_vf, 0.0);
        assert!(summary.pareto_vfs_in_lut.is_empty());
    }

    #[test]
    fn test_1_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(8, Shape::number());
        let lut1 = graph.add_lut(input1, FunctionTable::UNKWOWN);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        assert!(analysis.out_variances[lut1.i] == SymbolicVariance::LUT);
        assert!(analysis.out_shapes[lut1.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ZERO);
        assert_eq!(analysis.out_precisions[lut1.i], 8);
        assert_f64_eq(one_lut_cost, complexity_cost);
        let summary = analysis.noise_summary;
        assert!(summary.pareto_vfs_final.len() == 1);
        assert!(summary.pareto_vfs_in_lut.len() == 1);
        assert_f64_eq(summary.pareto_vfs_final[0].input_vf, 0.0);
        assert_f64_eq(summary.pareto_vfs_final[0].lut_vf, 1.0);
        assert_f64_eq(summary.pareto_vfs_in_lut[0].input_vf, 1.0);
        assert_f64_eq(summary.pareto_vfs_in_lut[0].lut_vf, 0.0);
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
            input_vf: norm2,
            lut_vf: 0.0,
        };
        assert!(analysis.out_variances[dot.i] == expected_var);
        assert!(analysis.out_shapes[dot.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION * 2);
        assert_eq!(analysis.out_precisions[dot.i], 1);
        let expected_dot_cost = (2 * lwe_dim) as f64;
        assert_f64_eq(expected_dot_cost, complexity_cost);
        let summary = analysis.noise_summary;
        assert!(summary.pareto_vfs_in_lut.is_empty());
        assert!(summary.pareto_vfs_final.len() == 1);
        assert_f64_eq(summary.pareto_vfs_final[0].input_vf, 5.0);
        assert_f64_eq(summary.pareto_vfs_final[0].lut_vf, 0.0);
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
        let actual_square_norm2 = analysis.out_variances[dot.i].input_vf;
        // Due to call on log2() to compute manp the result is not exact
        assert_f64_eq(actual_square_norm2, expected_square_norm2);
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION);
        assert_f64_eq(lwe_dim as f64, complexity_cost);
        let summary = analysis.noise_summary;
        assert!(summary.pareto_vfs_in_lut.is_empty());
        assert!(summary.pareto_vfs_final.len() == 1);
        assert_eq!(summary.pareto_vfs_final[0].origin(), VO::Input);
        assert_f64_eq(summary.pareto_vfs_final[0].input_vf, 5.0);
    }

    #[test]
    fn test_dot_tensorized_lut_dot_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::vector(2));
        let weights = &Weights::vector([1, 2]);
        let dot1 = graph.add_dot([input1], weights);
        let lut1 = graph.add_lut(dot1, FunctionTable::UNKWOWN);
        let dot2 = graph.add_dot([lut1, lut1], weights);
        let lut2 = graph.add_lut(dot2, FunctionTable::UNKWOWN);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        let expected_var_dot1 = SymbolicVariance {
            input_vf: weights.square_norm2() as f64,
            lut_vf: 0.0,
        };
        let expected_var_lut1 = SymbolicVariance {
            input_vf: 0.0,
            lut_vf: 1.0,
        };
        let expected_var_dot2 = SymbolicVariance {
            input_vf: 0.0,
            lut_vf: weights.square_norm2() as f64,
        };
        let expected_var_lut2 = SymbolicVariance {
            input_vf: 0.0,
            lut_vf: 1.0,
        };
        assert!(analysis.out_variances[dot1.i] == expected_var_dot1);
        assert!(analysis.out_variances[lut1.i] == expected_var_lut1);
        assert!(analysis.out_variances[dot2.i] == expected_var_dot2);
        assert!(analysis.out_variances[lut2.i] == expected_var_lut2);
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION * 4);
        let expected_cost = (lwe_dim * 4) as f64 + 2.0 * one_lut_cost;
        assert_f64_eq(expected_cost, complexity_cost);
        let summary = analysis.noise_summary;
        assert_eq!(summary.pareto_vfs_final.len(), 1);
        assert_eq!(summary.pareto_vfs_final[0].origin(), VO::Lut);
        assert_f64_eq(summary.pareto_vfs_final[0].lut_vf, 1.0);
        assert_eq!(summary.pareto_vfs_in_lut.len(), 1);
        assert_eq!(summary.pareto_vfs_in_lut[0].origin(), VO::Lut);
        assert_f64_eq(
            summary.pareto_vfs_in_lut[0].lut_vf,
            weights.square_norm2() as f64,
        );
    }

    #[test]
    fn test_lut_dot_mixed_lut() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let lut1 = graph.add_lut(input1, FunctionTable::UNKWOWN);
        let weights = &Weights::vector([2, 3]);
        let dot1 = graph.add_dot([input1, lut1], weights);
        let _lut2 = graph.add_lut(dot1, FunctionTable::UNKWOWN);
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity_cost(lwe_dim, one_lut_cost);

        let expected_cost = (2 * lwe_dim) as f64 + 2.0 * one_lut_cost;
        assert_f64_eq(expected_cost, complexity_cost);
        let expected_mixed = SymbolicVariance {
            input_vf: square(weights.values[0] as f64),
            lut_vf: square(weights.values[1] as f64),
        };
        let summary = analysis.noise_summary;
        assert_eq!(summary.pareto_vfs_final.len(), 1);
        assert_eq!(summary.pareto_vfs_final[0], SymbolicVariance::LUT);
        assert_eq!(summary.pareto_vfs_in_lut.len(), 1);
        assert_eq!(summary.pareto_vfs_in_lut[0].origin(), VO::Mixed);
        assert_eq!(summary.pareto_vfs_in_lut[0], expected_mixed);
    }
}
