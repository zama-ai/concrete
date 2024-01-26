use super::symbolic_variance::{SymbolicVariance, VarianceOrigin};
use crate::dag::operator::{
    dot_kind, DotKind, LevelledComplexity, OperatorIndex, Precision, Shape,
};
use crate::dag::rewrite::round::expand_round;
use crate::dag::unparametrized;
use crate::noise_estimator::error;
use crate::noise_estimator::p_error::{combine_errors, repeat_p_error};
use crate::optimization::config::NoiseBoundConfig;
use crate::utils::square;
use std::collections::{HashMap, HashSet};

// private short convention
use {DotKind as DK, VarianceOrigin as VO};
type Op = unparametrized::UnparameterizedOperator;

pub fn first<'a, Property>(inputs: &[OperatorIndex], properties: &'a [Property]) -> &'a Property {
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

fn assert_inputs_index(op: &unparametrized::UnparameterizedOperator, first_bad_index: usize) {
    let valid = match op {
        Op::Input { .. } => true,
        Op::Lut { input, .. } | Op::UnsafeCast { input, .. } | Op::Round { input, .. } => {
            input.i < first_bad_index
        }
        Op::LevelledOp { inputs, .. } | Op::Dot { inputs, .. } => {
            inputs.iter().all(|input| input.i < first_bad_index)
        }
    };
    assert!(valid, "Invalid dag, bad index in op: {op:?}");
}

fn assert_dag_correctness(dag: &unparametrized::OperationDag) {
    for (i, op) in dag.operators.iter().enumerate() {
        assert_non_empty_inputs(op);
        assert_inputs_uniform_precisions(op, &dag.out_precisions);
        assert_dot_uniform_inputs_shape(op, &dag.out_shapes);
        assert_inputs_index(op, i);
    }
}

pub fn has_round(dag: &unparametrized::OperationDag) -> bool {
    for op in &dag.operators {
        if matches!(op, Op::Round { .. }) {
            return true;
        }
    }
    false
}

pub fn has_unsafe_cast(dag: &unparametrized::OperationDag) -> bool {
    for op in &dag.operators {
        if matches!(op, Op::UnsafeCast { .. }) {
            return true;
        }
    }
    false
}

pub fn assert_no_round(dag: &unparametrized::OperationDag) {
    assert!(!has_round(dag));
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
    // All dominating final variance factor not entering a lut (usually final levelledOp)
    pub pareto_output: Vec<SymbolicVariance>,
    // All dominating variance factor entering a lut
    pub pareto_in_lut: Vec<SymbolicVariance>,
    // All counted variances for computing exact full dag error probability
    pub all_output: Vec<(u64, SymbolicVariance)>,
    pub all_in_lut: Vec<(u64, SymbolicVariance)>,
}

fn out_variance(
    op: &unparametrized::UnparameterizedOperator,
    out_shapes: &[Shape],
    out_variances: &[SymbolicVariance],
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
                DK::Simple | DK::Tensor | DK::Broadcast { .. } => {
                    let first_input = inputs[0];
                    let mut out_variance = SymbolicVariance::ZERO;
                    for (j, &weight) in weights.values.iter().enumerate() {
                        let k = if inputs.len() > 1 {
                            inputs[j].i
                        } else {
                            first_input.i
                        };
                        out_variance += out_variances[k] * square(weight as f64);
                    }
                    out_variance
                }
                DK::CompatibleTensor { .. } => todo!("TODO"),
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
        Op::UnsafeCast { input, .. } => out_variances[input.i],
        Op::Round { .. } => {
            unreachable!("Round should have been either expanded or integrated to a lut")
        }
    }
}

pub fn out_variances(dag: &unparametrized::OperationDag) -> Vec<SymbolicVariance> {
    let nb_ops = dag.operators.len();
    let mut out_variances = Vec::with_capacity(nb_ops);
    for op in &dag.operators {
        let vf = out_variance(op, &dag.out_shapes, &out_variances);
        out_variances.push(vf);
    }
    out_variances
}

pub fn extra_final_values_to_check(dag: &unparametrized::OperationDag) -> Vec<bool> {
    let nb_ops = dag.operators.len();
    let mut extra_values_to_check = vec![true; nb_ops];
    for op in &dag.operators {
        match op {
            Op::Input { .. } => (),
            Op::Lut { input, .. } | Op::UnsafeCast { input, .. } | Op::Round { input, .. } => {
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
) -> Vec<(Precision, Shape, SymbolicVariance)> {
    extra_final_values_to_check(dag)
        .iter()
        .enumerate()
        .filter_map(|(i, &is_final)| {
            if is_final {
                Some((
                    dag.out_precisions[i],
                    dag.out_shapes[i].clone(),
                    out_variances[i],
                ))
            } else {
                None
            }
        })
        .collect()
}

fn in_luts_variance(
    dag: &unparametrized::OperationDag,
    out_variances: &[SymbolicVariance],
) -> Vec<(Precision, Shape, SymbolicVariance)> {
    dag.operators
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            if let &Op::Lut { input, .. } = op {
                Some((
                    dag.out_precisions[input.i],
                    dag.out_shapes[i].clone(),
                    out_variances[input.i],
                ))
            } else {
                None
            }
        })
        .collect()
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
                DK::Simple | DK::Tensor | DK::Broadcast { .. } | DK::CompatibleTensor => {
                    LevelledComplexity::ADDITION * (inputs.len() as u64) * input_shape.flat_size()
                }
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
        Op::LevelledOp { complexity, .. } => *complexity,
        Op::Input { .. } | Op::Lut { .. } | Op::UnsafeCast { .. } => LevelledComplexity::ZERO,
        Op::Round { .. } => {
            unreachable!("Round should have been either expanded or integrated to a lut")
        }
    }
}

pub fn levelled_complexity(dag: &unparametrized::OperationDag) -> LevelledComplexity {
    let mut levelled_complexity = LevelledComplexity::ZERO;
    for op in &dag.operators {
        levelled_complexity += op_levelled_complexity(op, &dag.out_shapes);
    }
    levelled_complexity
}

pub fn lut_count_from_dag(dag: &unparametrized::OperationDag) -> u64 {
    let mut count = 0;
    for (i, op) in dag.operators.iter().enumerate() {
        if let Op::Lut { .. } = op {
            count += dag.out_shapes[i].flat_size();
        } else if let Op::Round { out_precision, .. } = op {
            count += dag.out_shapes[i].flat_size() * (dag.out_precisions[i] - out_precision) as u64;
        }
    }
    count
}

pub fn safe_noise_bound(precision: Precision, noise_config: &NoiseBoundConfig) -> f64 {
    error::safe_variance_bound_2padbits(
        precision as u64,
        noise_config.ciphertext_modulus_log,
        noise_config.maximum_acceptable_error_probability,
    )
}

fn constraints_by_precisions(
    out_precisions: &[Precision],
    final_variances: &[(Precision, Shape, SymbolicVariance)],
    in_luts_variance: &[(Precision, Shape, SymbolicVariance)],
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

fn select_precision<T1: Clone, T2: Copy>(
    target_precision: Precision,
    v: &[(Precision, T1, T2)],
) -> Vec<(T1, T2)> {
    v.iter()
        .filter_map(|(p, s, t)| {
            if *p == target_precision {
                Some((s.clone(), *t))
            } else {
                None
            }
        })
        .collect()
}

fn counted_symbolic_variance(
    symbolic_variances: &[(Shape, SymbolicVariance)],
) -> Vec<(u64, SymbolicVariance)> {
    pub fn exact_key(v: &SymbolicVariance) -> (u64, u64) {
        (v.lut_coeff.to_bits(), v.input_coeff.to_bits())
    }
    let mut count: HashMap<(u64, u64), u64> = HashMap::new();
    for (s, v) in symbolic_variances {
        *count.entry(exact_key(v)).or_insert(0) += s.flat_size();
    }
    let mut res = Vec::new();
    res.reserve_exact(count.len());
    for (_s, v) in symbolic_variances {
        if let Some(c) = count.remove(&exact_key(v)) {
            res.push((c, *v));
        }
    }
    res
}

fn constraint_for_one_precision(
    target_precision: Precision,
    extra_final_variances: &[(Precision, Shape, SymbolicVariance)],
    in_luts_variance: &[(Precision, Shape, SymbolicVariance)],
    safe_noise_bound: f64,
) -> VariancesAndBound {
    let extra_finals_variance = select_precision(target_precision, extra_final_variances);
    let in_luts_variance = select_precision(target_precision, in_luts_variance);
    let nb_luts = in_luts_variance.len() as u64;
    let all_output = counted_symbolic_variance(&extra_finals_variance);
    let all_in_lut = counted_symbolic_variance(&in_luts_variance);
    let remove_shape = |t: &(Shape, SymbolicVariance)| t.1;
    let extra_finals_variance = extra_finals_variance.iter().map(remove_shape).collect();
    let in_luts_variance = in_luts_variance.iter().map(remove_shape).collect();
    let pareto_vfs_final = SymbolicVariance::reduce_to_pareto_front(extra_finals_variance);
    let pareto_vfs_in_lut = SymbolicVariance::reduce_to_pareto_front(in_luts_variance);
    VariancesAndBound {
        precision: target_precision,
        safe_variance_bound: safe_noise_bound,
        nb_luts,
        pareto_output: pareto_vfs_final,
        pareto_in_lut: pareto_vfs_in_lut,
        all_output,
        all_in_lut,
    }
}

pub fn worst_log_norm_for_wop(dag: &unparametrized::OperationDag) -> f64 {
    assert_dag_correctness(dag);
    assert_no_round(dag);
    let out_variances = out_variances(dag);
    let in_luts_variance = in_luts_variance(dag, &out_variances);
    let coeffs = in_luts_variance
        .iter()
        .map(|(_precision, _shape, symbolic_variance)| {
            symbolic_variance.lut_coeff + symbolic_variance.input_coeff
        })
        .filter(|v| *v >= 1.0);
    let worst = coeffs.fold(1.0, f64::max);
    worst.log2()
}

pub fn analyze(
    dag: &unparametrized::OperationDag,
    noise_config: &NoiseBoundConfig,
) -> OperationDag {
    assert_dag_correctness(dag);
    let dag = &expand_round(dag);
    assert_no_round(dag);
    let out_variances = out_variances(dag);
    let in_luts_variance = in_luts_variance(dag, &out_variances);
    let nb_luts = lut_count_from_dag(dag);
    let extra_final_variances = extra_final_variances(dag, &out_variances);
    let levelled_complexity = levelled_complexity(dag);
    let constraints_by_precisions = constraints_by_precisions(
        &dag.out_precisions,
        &extra_final_variances,
        &in_luts_variance,
        noise_config,
    );
    let result = OperationDag {
        operators: dag.operators.clone(),
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

pub fn p_error_from_relative_variance(relative_variance: f64, kappa: f64) -> f64 {
    let sigma_scale = kappa / relative_variance.sqrt();
    error::error_probability_of_sigma_scale(sigma_scale)
}

fn p_error_per_constraint(
    constraint: &VariancesAndBound,
    input_noise_out: f64,
    blind_rotate_noise_out: f64,
    noise_keyswitch: f64,
    noise_modulus_switching: f64,
    kappa: f64,
) -> f64 {
    // Note: no log probability to keep accuracy near 0, 0 is a fine answer when p_success is very small.
    let mut p_error = 0.0;
    for &(count, vf) in &constraint.all_output {
        assert!(0 < count);
        let variance = vf.eval(input_noise_out, blind_rotate_noise_out);
        let relative_variance = variance / constraint.safe_variance_bound;
        let vf_p_error = p_error_from_relative_variance(relative_variance, kappa);

        p_error = combine_errors(p_error, repeat_p_error(vf_p_error, count));
    }
    // the maximal variance encountered during a lut computation
    for &(count, vf) in &constraint.all_in_lut {
        assert!(0 < count);
        let variance = vf.eval(input_noise_out, blind_rotate_noise_out);
        let relative_variance =
            (variance + noise_keyswitch + noise_modulus_switching) / constraint.safe_variance_bound;
        let vf_p_error = p_error_from_relative_variance(relative_variance, kappa);

        p_error = combine_errors(p_error, repeat_p_error(vf_p_error, count));
    }
    p_error
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
        let (relative_var, variance_bound) = peak_relative_variance(
            self,
            input_noise_out,
            blind_rotate_noise_out,
            noise_keyswitch,
            noise_modulus_switching,
        );
        (
            p_error_from_relative_variance(relative_var, kappa),
            relative_var * variance_bound,
        )
    }
    pub fn global_p_error(
        &self,
        input_noise_out: f64,
        blind_rotate_noise_out: f64,
        noise_keyswitch: f64,
        noise_modulus_switching: f64,
        kappa: f64,
    ) -> f64 {
        let mut p_error = 0.0;
        for ns in &self.constraints_by_precisions {
            let p_error_c = p_error_per_constraint(
                ns,
                input_noise_out,
                blind_rotate_noise_out,
                noise_keyswitch,
                noise_modulus_switching,
                kappa,
            );

            p_error = combine_errors(p_error, p_error_c);
        }
        assert!(0.0 <= p_error && p_error <= 1.0);
        p_error
    }

    pub fn feasible(
        &self,
        input_noise_out: f64,
        blind_rotate_noise_out: f64,
        noise_keyswitch: f64,
        noise_modulus_switching: f64,
    ) -> bool {
        for ns in &self.constraints_by_precisions {
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

    pub fn complexity(&self, input_lwe_dimension: u64, one_lut_cost: f64) -> f64 {
        let luts_cost = one_lut_cost * (self.nb_luts as f64);
        let levelled_cost = self.levelled_complexity.cost(input_lwe_dimension);
        luts_cost + levelled_cost
    }

    pub fn levelled_complexity(&self, input_lwe_dimension: u64) -> f64 {
        self.levelled_complexity.cost(input_lwe_dimension)
    }
}

#[cfg(test)]
pub mod tests {

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

    pub const CONFIG: NoiseBoundConfig = NoiseBoundConfig {
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
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

        assert_eq!(analysis.out_variances[input1.i], SymbolicVariance::INPUT);
        assert_eq!(graph.out_shapes[input1.i], Shape::number());
        assert_eq!(analysis.levelled_complexity, LevelledComplexity::ZERO);
        assert_eq!(graph.out_precisions[input1.i], 1);
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
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

        assert!(analysis.out_variances[lut1.i] == SymbolicVariance::LUT);
        assert!(graph.out_shapes[lut1.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ZERO);
        assert_eq!(graph.out_precisions[lut1.i], 8);
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
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

        let expected_var = SymbolicVariance {
            input_coeff: norm2,
            lut_coeff: 0.0,
        };
        assert!(analysis.out_variances[dot.i] == expected_var);
        assert!(graph.out_shapes[dot.i] == Shape::number());
        assert!(analysis.levelled_complexity == LevelledComplexity::ADDITION * 2);
        assert_eq!(graph.out_precisions[dot.i], 1);
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
        #[allow(clippy::imprecise_flops)]
        let manp = (1.0 * 1.0 + 2.0 * 2_f64).sqrt();
        let dot = graph.add_levelled_op([input1, input1], cpx_dot, manp, Shape::number(), "dot");
        let analysis = analyze(&graph);
        let one_lut_cost = 100.0;
        let lwe_dim = 1024;
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

        assert!(analysis.out_variances[dot.i].origin() == VO::Input);
        assert_eq!(graph.out_precisions[dot.i], 3);
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
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

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
        let complexity_cost = analysis.complexity(lwe_dim, one_lut_cost);

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
            _ = graph.add_input(i, Shape::number());
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

    #[test]
    fn test_broadcast_dot_multiply_by_number() {
        let mut graph = unparametrized::OperationDag::new();
        let shape = Shape {
            dimensions_size: vec![2, 2],
        };
        let input1 = graph.add_input(1, &shape);
        let weights = &Weights::number(2);
        _ = graph.add_dot([input1], weights);
        assert!(*graph.out_shapes.last().unwrap() == shape);
        let analysis = analyze(&graph);
        assert_f64_eq(analysis.out_variances.last().unwrap().input_coeff, 4.0);
    }

    #[test]
    fn test_broadcast_dot_add_tensor() {
        let mut graph = unparametrized::OperationDag::new();
        let shape = Shape {
            dimensions_size: vec![2, 2],
        };
        let input1 = graph.add_input(1, &shape);
        let input2 = graph.add_input(1, &shape);
        let lut2 = graph.add_lut(input2, FunctionTable::UNKWOWN, 1);
        let weights = &Weights::vector([2, 3]);
        _ = graph.add_dot([input1, lut2], weights);
        assert!(*graph.out_shapes.last().unwrap() == shape);
        let analysis = analyze(&graph);
        assert_f64_eq(analysis.out_variances.last().unwrap().input_coeff, 4.0);
        assert_f64_eq(analysis.out_variances.last().unwrap().lut_coeff, 9.0);
    }
}
