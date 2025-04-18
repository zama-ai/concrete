use std::ops::{Add, Deref, Index, IndexMut};

use crate::dag::operator::{DotKind, LevelledComplexity, Operator, OperatorIndex, Precision};
use crate::dag::rewrite::round::expand_round_and_index_map;
use crate::dag::unparametrized::{Dag, DagOperator};
use crate::optimization::config::NoiseBoundConfig;
use crate::optimization::dag::multi_parameters::partition_cut::PartitionCut;
use crate::optimization::dag::multi_parameters::partitionning::partitionning_with_preferred;
use crate::optimization::dag::multi_parameters::partitions::{
    InstructionPartition, PartitionIndex, Transition,
};
use crate::optimization::dag::solo_key::analyze::safe_noise_bound;
use crate::optimization::{Err, Result};

use super::complexity::OperationsCount;
use super::keys_spec;
use super::noise_expression::{
    bootstrap_noise, fast_keyswitch_noise, input_noise, keyswitch_noise, modulus_switching_noise,
    NoiseExpression,
};
use super::partitions::Partitions;
use super::symbolic::{bootstrap, fast_keyswitch, keyswitch, SymbolMap};
use super::variance_constraint::VarianceConstraint;
use crate::utils::square;

const MAX_FORWARDING: u16 = 500;

#[derive(Debug, Clone)]
pub struct PartitionedDag {
    pub(crate) dag: Dag,
    pub(crate) partitions: Partitions,
}

impl PartitionedDag {
    fn get_initial_variances(&self) -> Variances {
        let vars = self
            .dag
            .get_operators_iter()
            .map(|op| match op.operator {
                Operator::Input { .. } => {
                    let mut output = OperatorVariance::zero(self.partitions.nb_partitions);
                    let op_partition = self.partitions[op.id].instruction_partition;
                    output[op_partition] += 1.0 * input_noise(op_partition);
                    output
                }
                _ => OperatorVariance::zero(self.partitions.nb_partitions),
            })
            .collect();
        Variances { vars }
    }
}

#[derive(Debug)]
pub struct VariancedDagOperator<'a> {
    dag: &'a VariancedDag,
    id: OperatorIndex,
}

impl VariancedDagOperator<'_> {
    #[allow(unused)]
    fn get_inputs_iter(&self) -> impl Iterator<Item = VariancedDagOperator<'_>> {
        self.operator()
            .get_inputs_iter()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|n| self.dag.get_operator(n.id))
    }

    pub(crate) fn operator(&self) -> DagOperator<'_> {
        self.dag.dag.get_operator(self.id)
    }

    #[allow(unused)]
    pub(crate) fn partition(&self) -> &InstructionPartition {
        &self.dag.partitions[self.id]
    }

    pub(crate) fn variance(&self) -> &OperatorVariance {
        &self.dag.variances[self.id]
    }
}

pub struct VariancedDagOperatorMut<'a> {
    dag: &'a mut VariancedDag,
    id: OperatorIndex,
}

impl VariancedDagOperatorMut<'_> {
    fn get_inputs_iter(&self) -> impl Iterator<Item = VariancedDagOperator<'_>> {
        self.operator()
            .get_inputs_iter()
            .collect::<Vec<_>>()
            .into_iter()
            .map(|n| self.dag.get_operator(n.id))
    }

    pub(crate) fn operator(&self) -> DagOperator<'_> {
        self.dag.dag.get_operator(self.id)
    }

    pub(crate) fn partition(&self) -> &InstructionPartition {
        &self.dag.partitions[self.id]
    }

    #[allow(unused)]
    pub(crate) fn partition_mut(&mut self) -> &mut InstructionPartition {
        &mut self.dag.partitions[self.id]
    }

    #[allow(unused)]
    pub(crate) fn variance(&self) -> &OperatorVariance {
        &self.dag.variances[self.id]
    }

    pub(crate) fn variance_mut(&mut self) -> &mut OperatorVariance {
        &mut self.dag.variances[self.id]
    }
}

#[derive(Debug, Clone)]
pub struct VariancedDag {
    pub(crate) dag: Dag,
    pub(crate) partitions: Partitions,
    pub(crate) variances: Variances,
    pub(crate) external_variance_constraints: Vec<VarianceConstraint>,
}

impl VariancedDag {
    fn try_from_partitioned(partitioned: PartitionedDag) -> Result<Self> {
        // We compute the initial variances with noise at input nodes and NANs everywhere
        // else.
        let variances = partitioned.get_initial_variances();
        let PartitionedDag { dag, partitions } = partitioned;
        let mut varianced = Self {
            dag,
            partitions,
            variances,
            external_variance_constraints: vec![],
        };

        // We forward the noise once to verify the composability.
        varianced.apply_external_partition_input_variance();
        let _ = varianced.forward_noise();
        varianced.check_composability()?;
        varianced.apply_composition_rules();

        // We loop, forwarding the noise, until it settles.
        for _ in 0..MAX_FORWARDING {
            // The noise gets computed from inputs down to outputs.
            if varianced.forward_noise() {
                // Noise settled, we return the varianced dag.
                varianced.collect_external_input_constraint();
                varianced.collect_external_output_constraint();
                return Ok(varianced);
            }
            // The noise of the inputs gets updated following the composition rules
            varianced.apply_composition_rules();
        }

        panic!("Forwarding of noise did not reach a fixed point.")
    }

    fn get_operator(&self, index: OperatorIndex) -> VariancedDagOperator<'_> {
        VariancedDagOperator {
            dag: self,
            id: index,
        }
    }

    fn get_operator_mut(&mut self, index: OperatorIndex) -> VariancedDagOperatorMut<'_> {
        VariancedDagOperatorMut {
            dag: self,
            id: index,
        }
    }

    /// Patches the inputs following the composition rules.
    fn apply_composition_rules(&mut self) {
        for (to, froms) in self.dag.composition.clone() {
            let maxed_variance = froms
                .into_iter()
                .map(|id| self.get_operator(id).variance().to_owned())
                .reduce(|acc, var| acc.partition_wise_max(&var))
                .unwrap();
            let mut input = self.get_operator_mut(to);
            *(input.variance_mut()) = maxed_variance;
        }
    }

    fn apply_external_partition_input_variance(&mut self) {
        let p_cut = self.partitions.p_cut.clone();
        for (i, op) in self.dag.operators.clone().iter().enumerate() {
            if let Operator::Input { .. } = op {
                let partition_index = self.partitions.instrs_partition[i].instruction_partition;
                if p_cut.is_external_partition(&partition_index) {
                    let external_partition =
                        &p_cut.external_partitions[p_cut.external_partition_index(partition_index)];
                    let max_variance = external_partition.max_variance;
                    let variance = external_partition.variance;

                    let mut input = self.get_operator_mut(OperatorIndex(i));
                    let mut variances = input.variance().clone();
                    variances.vars[partition_index.0] = NoiseExpression::zero()
                        + (max_variance / variance) * bootstrap_noise(partition_index);
                    *(input.variance_mut()) = variances;
                }
            }
        }
    }

    fn collect_external_input_constraint(&mut self) {
        let p_cut = &self.partitions.p_cut;
        for op in self.dag.get_operators_iter() {
            if let Operator::Input {
                out_precision,
                out_shape,
            } = op.operator
            {
                let partition_index =
                    self.partitions.instrs_partition[op.id.0].instruction_partition;
                if !p_cut.is_external_partition(&partition_index) {
                    continue;
                }

                let max_variance = p_cut.external_partitions
                    [p_cut.external_partition_index(partition_index)]
                .max_variance;

                let variances = &self.get_operator(op.id).variance().vars.clone();
                for variance in variances.iter() {
                    let constraint = VarianceConstraint {
                        precision: *out_precision,
                        nb_partitions: self.partitions.nb_partitions,
                        partition: partition_index,
                        nb_constraints: out_shape.flat_size(),
                        safe_variance_bound: max_variance,
                        noise_expression: variance.clone(),
                        noise_evaluator: None,
                        location: op.location.clone(),
                    };
                    self.external_variance_constraints.push(constraint);
                }
            }
        }
    }

    fn collect_external_output_constraint(&mut self) {
        let p_cut = self.partitions.p_cut.clone();
        for dag_op in self.dag.get_output_operators_iter() {
            let DagOperator {
                id: op_index,
                shape: out_shape,
                precision: out_precision,
                ..
            } = dag_op;
            let optional_partition_index = p_cut.partition(&self.dag, op_index);
            if optional_partition_index.is_none() {
                continue;
            }
            let partition_index = optional_partition_index.unwrap();
            if !p_cut.is_external_partition(&partition_index) {
                continue;
            }
            let max_variance = p_cut.external_partitions
                [p_cut.external_partition_index(partition_index)]
            .max_variance;

            let variances = &self.get_operator(op_index).variance().vars.clone();
            for variance in variances.iter() {
                let constraint = VarianceConstraint {
                    precision: *out_precision,
                    nb_partitions: self.partitions.nb_partitions,
                    partition: partition_index,
                    nb_constraints: out_shape.flat_size(),
                    safe_variance_bound: max_variance,
                    noise_expression: variance.clone(),
                    noise_evaluator: None,
                    location: dag_op.location.clone(),
                };
                self.external_variance_constraints.push(constraint);
            }
        }
    }

    /// Propagates the noise downward in the graph.
    fn forward_noise(&mut self) -> bool {
        // We save the old variance to compute the diff at the end.
        let old_variances = self.variances.clone();
        let nb_partitions = self.partitions.nb_partitions;

        // We loop through the operators and propagate the noise.
        for operator_id in self.dag.get_indices_iter() {
            let mut operator = self.get_operator_mut(operator_id);
            // Inputs are already computed
            if operator.operator().is_input() {
                continue;
            }

            // Operator variance will be used to override the noise
            let mut operator_variance = OperatorVariance::zero(nb_partitions);
            let operator_partition = operator.partition().instruction_partition;

            // We first compute the noise in the partition of the operator
            operator_variance[operator.partition().instruction_partition] = match operator
                .operator()
                .operator
            {
                Operator::Lut { .. } => {
                    NoiseExpression::zero() + 1.0 * bootstrap_noise(operator_partition)
                }
                Operator::MaxNoise { .. } => {
                    operator
                        .get_inputs_iter()
                        .fold(NoiseExpression::zero(), |acc, inp| {
                            let inp_noise = inp.variance()[operator_partition].clone();
                            NoiseExpression::max(&acc, &inp_noise)
                        })
                }
                Operator::LinearNoise { weights, .. } => operator
                    .get_inputs_iter()
                    .zip(weights)
                    .fold(NoiseExpression::zero(), |acc, (inp, &weight)| {
                        let inp_noise = inp.variance()[operator_partition].clone();
                        acc + inp_noise * square(weight)
                    }),
                Operator::Dot {
                    weights,
                    kind: DotKind::Simple | DotKind::Tensor | DotKind::Broadcast { .. },
                    ..
                } => weights
                    .values
                    .iter()
                    .zip(operator.get_inputs_iter().map(|n| n.variance().clone()))
                    .fold(NoiseExpression::zero(), |acc, (weight, var)| {
                        let inp_var = var[operator_partition].clone();
                        acc + inp_var * square(*weight as f64)
                    }),
                Operator::UnsafeCast { .. } | Operator::ChangePartition { .. } => {
                    operator.get_inputs_iter().next().unwrap().variance()[operator_partition]
                        .clone()
                }
                Operator::Input { .. } | Operator::ZeroNoise { .. } => unreachable!(),

                Operator::Dot {
                    kind: DotKind::CompatibleTensor,
                    ..
                } => todo!("TODO"),
                Operator::Dot {
                    kind: DotKind::Unsupported,
                    ..
                } => panic!("Unsupported"),
                Operator::Round { .. } => {
                    unreachable!("Round should have been either expanded or integrated to a lut")
                }
            };
            // We add the noise for the transitions to alternative representations
            operator
                .partition()
                .alternative_output_representation
                .iter()
                .for_each(|index| {
                    let noise_in_operator_partition = operator_variance[operator_partition].clone();
                    operator_variance[*index] = noise_in_operator_partition
                        + 1.0 * fast_keyswitch_noise(operator_partition, *index);
                });
            // We override the noise
            *operator.variance_mut() = operator_variance;
        }

        // We return whether there is a diff or not.
        old_variances == self.variances
    }

    #[allow(unused)]
    fn check_composability(&self) -> Result<()> {
        self.dag
            .composition
            .clone()
            .into_iter()
            .flat_map(|(to, froms)| froms.into_iter())
            .map(|i| self.get_operator(i))
            .filter(|op| op.operator().is_output())
            .try_for_each(|op| {
                let id = op.id;
                let loc = op.operator().location;
                op.variance()
                    .check_growing_input_noise()
                    .map_err(|err| match err {
                        Err::NotComposable(prev) => {
                            Err::NotComposable(format!("At location {loc}:\n{prev}."))
                        }
                        _ => unreachable!(),
                    })
            })
    }
}

#[derive(Debug)]
pub struct AnalyzedDag {
    pub operators: Vec<Operator>,
    // Collect all operators ouput variances
    pub nb_partitions: usize,
    pub instrs_partition: Vec<InstructionPartition>,
    pub instrs_variances: Vec<OperatorVariance>,
    // The full dag levelled complexity
    pub levelled_complexity: LevelledComplexity,
    // All variance constraints including dominated ones
    pub variance_constraints: Vec<VarianceConstraint>,
    // Undominated variance constraints
    pub undominated_variance_constraints: Vec<VarianceConstraint>,
    pub operations_count_per_instrs: Vec<OperationsCount>,
    pub operations_count: OperationsCount,
    pub instruction_rewrite_index: Vec<Vec<OperatorIndex>>,
    pub p_cut: PartitionCut,
}

pub fn analyze(
    dag: &Dag,
    noise_config: &NoiseBoundConfig,
    p_cut: &Option<PartitionCut>,
    default_partition: PartitionIndex,
) -> Result<AnalyzedDag> {
    let (dag, instruction_rewrite_index) = expand_round_and_index_map(dag);
    let levelled_complexity = LevelledComplexity::ZERO;
    // The precision cut is chosen to work well with rounded pbs
    // Note: this is temporary
    #[allow(clippy::option_if_let_else)]
    let p_cut = match p_cut {
        Some(p_cut) => p_cut.clone(),
        None => PartitionCut::for_each_precision(&dag),
    };
    let partitions = partitionning_with_preferred(&dag, &p_cut, default_partition);
    let partitioned_dag = PartitionedDag { dag, partitions };
    let varianced_dag = VariancedDag::try_from_partitioned(partitioned_dag)?;
    let mut variance_constraints = collect_all_variance_constraints(&varianced_dag, noise_config);
    // add external variance constraints
    variance_constraints.extend_from_slice(varianced_dag.external_variance_constraints.as_slice());
    let undominated_variance_constraints =
        VarianceConstraint::remove_dominated(&variance_constraints);
    let operations_count_per_instrs = collect_operations_count(&varianced_dag);
    let operations_count = operations_count_per_instrs
        .clone()
        .into_iter()
        .reduce(Add::add)
        .unwrap();
    Ok(AnalyzedDag {
        operators: varianced_dag.dag.operators,
        instruction_rewrite_index,
        nb_partitions: varianced_dag.partitions.nb_partitions,
        instrs_partition: varianced_dag.partitions.instrs_partition,
        instrs_variances: varianced_dag.variances.vars,
        levelled_complexity,
        variance_constraints,
        undominated_variance_constraints,
        operations_count_per_instrs,
        operations_count,
        p_cut,
    })
}

pub fn original_instrs_partition(
    dag: &AnalyzedDag,
    keys: &keys_spec::ExpandedCircuitKeys,
) -> Vec<keys_spec::InstructionKeys> {
    let big_keys = &keys.big_secret_keys;
    let ks_keys = &keys.keyswitch_keys;
    let pbs_keys = &keys.bootstrap_keys;
    let fks_keys = &keys.conversion_keyswitch_keys;
    let mut result = vec![];
    result.reserve_exact(dag.instruction_rewrite_index.len());
    let unknown = keys_spec::Id::MAX;
    for new_instructions in &dag.instruction_rewrite_index {
        let mut partition = None;
        let mut input_partition = None;
        let mut tlu_keyswitch_key = None;
        let mut tlu_bootstrap_key = None;
        let mut conversion_key = None;
        // let mut extra_conversion_keys = None;
        for (i, new_instruction) in new_instructions.iter().enumerate() {
            // focus on TLU information
            let new_instr_part = &dag.instrs_partition[new_instruction.0];
            if let Operator::Lut { .. } = dag.operators[new_instruction.0] {
                let ks_dst = new_instr_part.instruction_partition;
                partition = Some(ks_dst);
                #[allow(clippy::match_on_vec_items)]
                let ks_src = match new_instr_part.inputs_transition[0] {
                    Some(Transition::Internal { src_partition }) => src_partition,
                    None => ks_dst,
                    _ => unreachable!(),
                };
                input_partition = Some(ks_src);
                let ks_key = ks_keys[ks_src.0][ks_dst.0].as_ref().unwrap().identifier;
                let pbs_key = pbs_keys[ks_dst.0].identifier;
                assert!(tlu_keyswitch_key.unwrap_or(ks_key) == ks_key);
                assert!(tlu_bootstrap_key.unwrap_or(pbs_key) == pbs_key);
                tlu_keyswitch_key = Some(ks_key);
                tlu_bootstrap_key = Some(pbs_key);
            }
            if !new_instr_part.alternative_output_representation.is_empty() {
                assert!(new_instr_part.alternative_output_representation.len() == 1);
                let src = new_instr_part.instruction_partition;
                let dst = *new_instr_part
                    .alternative_output_representation
                    .iter()
                    .next()
                    .unwrap();
                let key = fks_keys[src.0][dst.0].as_ref().unwrap().identifier;
                assert!(conversion_key.unwrap_or(key) == key);
                conversion_key = Some(key);
            }
            // Only last instruction can have alternative conversion
            assert!(
                new_instr_part.alternative_output_representation.is_empty()
                    || i == new_instructions.len() - 1
            );
        }
        let partition =
            partition.unwrap_or(dag.instrs_partition[new_instructions[0].0].instruction_partition);
        let input_partition = input_partition.unwrap_or(partition);
        let merged = keys_spec::InstructionKeys {
            input_key: big_keys[input_partition.0].identifier,
            tlu_keyswitch_key: tlu_keyswitch_key.unwrap_or(unknown),
            tlu_bootstrap_key: tlu_bootstrap_key.unwrap_or(unknown),
            output_key: big_keys[partition.0].identifier,
            extra_conversion_keys: conversion_key.iter().copied().collect(),
            tlu_circuit_bootstrap_key: keys_spec::NO_KEY_ID,
            tlu_private_functional_packing_key: keys_spec::NO_KEY_ID,
        };
        result.push(merged);
    }
    result
}

#[derive(PartialEq, Debug, Clone)]
pub struct OperatorVariance {
    pub(crate) vars: Vec<NoiseExpression>,
}

impl Index<PartitionIndex> for OperatorVariance {
    type Output = NoiseExpression;

    fn index(&self, index: PartitionIndex) -> &Self::Output {
        &self.vars[index.0]
    }
}

impl IndexMut<PartitionIndex> for OperatorVariance {
    fn index_mut(&mut self, index: PartitionIndex) -> &mut Self::Output {
        &mut self.vars[index.0]
    }
}

impl Deref for OperatorVariance {
    type Target = [NoiseExpression];

    fn deref(&self) -> &Self::Target {
        &self.vars
    }
}

impl OperatorVariance {
    pub fn zero(nb_partitions: usize) -> Self {
        Self {
            vars: (0..nb_partitions)
                .map(|_| NoiseExpression::zero())
                .collect(),
        }
    }

    pub fn nb_partitions(&self) -> usize {
        self.vars.len()
    }

    pub fn partition_wise_max(&self, other: &Self) -> Self {
        let vars = self
            .vars
            .iter()
            .zip(other.vars.iter())
            .map(|(s, o)| NoiseExpression::max(s, o))
            .collect();
        Self { vars }
    }

    pub fn check_growing_input_noise(&self) -> Result<()> {
        self.vars
            .iter()
            .enumerate()
            .flat_map(|(var_i, var)| {
                PartitionIndex::range(0, self.nb_partitions())
                    .map(move |part_i| (var_i, part_i, var.coeff(input_noise(part_i))))
            })
            .try_for_each(|(var, partition, coeff)| {
                if coeff > 1.0 {
                    Result::Err(Err::NotComposable(format!(
                        "The noise of the node {var} is contaminated by noise coming straight from the input (partition: {partition}, coeff: {coeff:.2})"
                    )))
                } else {
                    Ok(())
                }
            })
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Variances {
    pub(crate) vars: Vec<OperatorVariance>,
}

impl Index<OperatorIndex> for Variances {
    type Output = OperatorVariance;

    fn index(&self, index: OperatorIndex) -> &Self::Output {
        &self.vars[index.0]
    }
}

impl IndexMut<OperatorIndex> for Variances {
    fn index_mut(&mut self, index: OperatorIndex) -> &mut Self::Output {
        &mut self.vars[index.0]
    }
}

impl Deref for Variances {
    type Target = [OperatorVariance];

    fn deref(&self) -> &Self::Target {
        &self.vars
    }
}

#[allow(unused)]
fn variance_constraint(
    dag: &Dag,
    noise_config: &NoiseBoundConfig,
    nb_partitions: usize,
    partition: PartitionIndex,
    op_i: usize,
    precision: Precision,
    noise: NoiseExpression,
) -> VarianceConstraint {
    let nb_constraints = dag.out_shapes[op_i].flat_size();
    let safe_variance_bound = safe_noise_bound(precision, noise_config);
    let location = dag.locations[op_i].clone();
    VarianceConstraint {
        precision,
        partition,
        nb_constraints,
        safe_variance_bound,
        nb_partitions,
        noise_expression: noise,
        noise_evaluator: None,
        location,
    }
}

#[allow(unused)]
#[allow(clippy::float_cmp)]
#[allow(clippy::match_on_vec_items)]
fn collect_all_variance_constraints(
    dag: &VariancedDag,
    noise_config: &NoiseBoundConfig,
) -> Vec<VarianceConstraint> {
    let VariancedDag {
        dag,
        partitions,
        variances,
        ..
    } = dag;
    let mut constraints = vec![];
    for op in dag.get_operators_iter() {
        let partition = partitions[op.id].instruction_partition;
        if let Operator::Lut { input, .. } = op.operator {
            let precision = dag.out_precisions[input.0];
            let dst_partition = partition;
            let src_partition = match partitions[op.id].inputs_transition[0] {
                None => dst_partition,
                Some(Transition::Internal { src_partition }) => {
                    assert!(src_partition != dst_partition);
                    src_partition
                }
                Some(Transition::Additional { src_partition }) => {
                    assert!(src_partition != dst_partition);
                    let variance = &variances[*input][dst_partition];
                    assert!(
                        variance.coeff(fast_keyswitch_noise(src_partition, dst_partition)) == 1.0
                    );
                    dst_partition
                }
            };
            let variance = variances[*input][src_partition].clone();
            let variance = variance
                + 1.0 * keyswitch_noise(src_partition, dst_partition)
                + 1.0 * modulus_switching_noise(partition);
            constraints.push(variance_constraint(
                dag,
                noise_config,
                partitions.nb_partitions,
                partition,
                op.id.0,
                precision,
                variance,
            ));
        }
        if op.is_output() {
            let precision = dag.out_precisions[op.id.0];
            let variance = variances[op.id][partition].clone();
            constraints.push(variance_constraint(
                dag,
                noise_config,
                partitions.nb_partitions,
                partition,
                op.id.0,
                precision,
                variance,
            ));
        }
    }
    constraints
}

#[allow(unused)]
#[allow(clippy::match_on_vec_items)]
fn operations_counts(
    dag: &Dag,
    op: &Operator,
    nb_partitions: usize,
    instr_partition: &InstructionPartition,
) -> OperationsCount {
    let mut counts = SymbolMap::new();
    if let Operator::Lut { input, .. } = op {
        let partition = instr_partition.instruction_partition;
        let nb_lut = dag.out_shapes[input.0].flat_size() as usize;
        let src_partition = match instr_partition.inputs_transition[0] {
            Some(Transition::Internal { src_partition }) => src_partition,
            Some(Transition::Additional { .. }) | None => partition,
        };
        counts.update(keyswitch(src_partition, partition), |a| a + nb_lut);
        counts.update(bootstrap(partition), |a| a + nb_lut);
        for &conv_partition in &instr_partition.alternative_output_representation {
            counts.update(fast_keyswitch(partition, conv_partition), |a| a + nb_lut);
        }
    }
    OperationsCount(counts)
}

#[allow(unused)]
fn collect_operations_count(dag: &VariancedDag) -> Vec<OperationsCount> {
    dag.dag
        .operators
        .iter()
        .enumerate()
        .map(|(i, op)| {
            operations_counts(
                &dag.dag,
                op,
                dag.partitions.nb_partitions,
                &dag.partitions[OperatorIndex(i)],
            )
        })
        .collect()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dag::operator::{FunctionTable, Shape};
    use crate::dag::unparametrized;
    use crate::optimization::dag::multi_parameters::partitionning::tests::{
        show_partitionning, HIGH_PRECISION_PARTITION, LOW_PRECISION_PARTITION,
    };
    use crate::optimization::dag::solo_key::analyze::tests::CONFIG;

    pub fn analyze(dag: &unparametrized::Dag) -> AnalyzedDag {
        analyze_with_preferred(dag, LOW_PRECISION_PARTITION)
    }

    pub fn analyze_with_preferred(
        dag: &unparametrized::Dag,
        default_partition: PartitionIndex,
    ) -> AnalyzedDag {
        let p_cut = PartitionCut::for_each_precision(dag);
        super::analyze(dag, &CONFIG, &Some(p_cut), default_partition).unwrap()
    }

    #[allow(clippy::float_cmp)]
    fn assert_input_on(
        dag: &AnalyzedDag,
        partition: PartitionIndex,
        op_i: usize,
        expected_coeff: f64,
    ) {
        for symbolic_variance_partition in [LOW_PRECISION_PARTITION, HIGH_PRECISION_PARTITION] {
            let sb = dag.instrs_variances[op_i][partition].clone();
            let coeff = sb.coeff(input_noise(symbolic_variance_partition));
            if symbolic_variance_partition == partition {
                assert!(
                    coeff == expected_coeff,
                    "INCORRECT INPUT COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.instrs_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            } else {
                assert!(
                    coeff == 0.0,
                    "INCORRECT INPUT COEFF ON WRONG PARTITION {:?} {:?} {} {}",
                    dag.instrs_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            }
        }
    }

    #[allow(clippy::float_cmp)]
    fn assert_pbs_on(
        dag: &AnalyzedDag,
        partition: PartitionIndex,
        op_i: usize,
        expected_coeff: f64,
    ) {
        for symbolic_variance_partition in [LOW_PRECISION_PARTITION, HIGH_PRECISION_PARTITION] {
            let sb = dag.instrs_variances[op_i][partition].clone();
            eprintln!("{:?}", dag.instrs_variances[op_i]);
            eprintln!("{:?}", dag.instrs_variances[op_i][partition]);
            let coeff = sb.coeff(bootstrap_noise(symbolic_variance_partition));
            if symbolic_variance_partition == partition {
                assert!(
                    coeff == expected_coeff,
                    "INCORRECT PBS COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.instrs_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            } else {
                assert!(
                    coeff == 0.0,
                    "INCORRECT PBS COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.instrs_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Forwarding of noise did not reach a fixed point.")]
    fn test_decreasing_panics() {
        let mut dag = unparametrized::Dag::new();
        let inp = dag.add_input(1, Shape::number());
        let oup = dag.add_linear_noise(
            [inp],
            LevelledComplexity::ZERO,
            [0.5],
            Shape::number(),
            "comment",
        );
        dag.add_composition(oup, inp);
        let p_cut = PartitionCut::for_each_precision(&dag);
        let _ = super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION).unwrap();
    }

    #[test]
    fn test_composition_with_nongrowing_inputs_only() {
        let mut dag = unparametrized::Dag::new();
        let inp = dag.add_input(1, Shape::number());
        let oup = dag.add_linear_noise(
            [inp],
            LevelledComplexity::ZERO,
            [1.0],
            Shape::number(),
            "comment",
        );
        dag.add_composition(oup, inp);
        let p_cut = PartitionCut::for_each_precision(&dag);
        let analyzed_dag =
            super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION).unwrap();
        let last_var = analyzed_dag.instrs_variances[analyzed_dag.instrs_variances.len() - 1]
            [PartitionIndex(0)]
        .to_string();
        assert_eq!(last_var, "1σ²In[0]");
    }

    #[test]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: NotComposable(\"At location unknown:\\nThe noise of the node 0 is contaminated by noise coming straight from the input (partition: 0, coeff: 1.21).\")"
    )]
    fn test_composition_with_growing_inputs_panics() {
        let mut dag = unparametrized::Dag::new();
        let inp = dag.add_input(1, Shape::number());
        let oup = dag.add_linear_noise(
            [inp],
            LevelledComplexity::ZERO,
            [1.1],
            Shape::number(),
            "comment",
        );
        dag.add_composition(oup, inp);
        let p_cut = PartitionCut::for_each_precision(&dag);
        let _ = super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION).unwrap();
    }

    #[test]
    fn test_composition_1_partition() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(1, Shape::number());
        let output = dag.add_lut(input1, FunctionTable::UNKWOWN, 2);
        dag.add_composition(output, input1);
        let p_cut = PartitionCut::for_each_precision(&dag);
        let dag = super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION).unwrap();
        assert!(dag.nb_partitions == 1);
        let actual_constraint_strings = dag
            .variance_constraints
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let expected_constraint_strings = vec![
            "1σ²Br[0] + 1σ²K[0] + 1σ²M[0] < (2²)**-5 (1bits partition:0 count:1, dom=10)",
            "1σ²Br[0] < (2²)**-6 (2bits partition:0 count:1, dom=12)",
        ];
        assert!(actual_constraint_strings == expected_constraint_strings);
    }

    #[test]
    fn test_composition_2_partitions() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(3, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 6);
        let lut3 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 3);
        let input2 = dag.add_dot([input1, lut3], [1, 1]);
        let output = dag.add_lut(input2, FunctionTable::UNKWOWN, 3);
        dag.add_compositions([output], [input1]);
        let analyzed_dag = super::analyze(&dag, &CONFIG, &None, LOW_PRECISION_PARTITION).unwrap();
        assert_eq!(analyzed_dag.nb_partitions, 2);
        let actual_constraint_strings = analyzed_dag
            .variance_constraints
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let expected_constraint_strings = vec![
            "1σ²Br[0] + 1σ²K[0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] + 1σ²K[0→1] + 1σ²M[1] < (2²)**-10 (6bits partition:1 count:1, dom=20)",
            "1σ²Br[0] + 1σ²Br[1] + 1σ²K[0] + 1σ²FK[1→0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
        ];
        assert_eq!(actual_constraint_strings, expected_constraint_strings);
        let partitions = vec![
            LOW_PRECISION_PARTITION,
            LOW_PRECISION_PARTITION,
            HIGH_PRECISION_PARTITION,
            LOW_PRECISION_PARTITION,
            LOW_PRECISION_PARTITION,
        ];
        assert_eq!(
            partitions,
            analyzed_dag
                .instrs_partition
                .iter()
                .map(|p| p.instruction_partition)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_composition_3_partitions() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(3, Shape::number());
        let input2 = dag.add_input(13, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 6);
        let lut3 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 3);
        let a = dag.add_dot([input2, lut3], [1, 1]);
        let b = dag.add_dot([input1, lut3], [1, 1]);
        let out1 = dag.add_lut(a, FunctionTable::UNKWOWN, 3);
        let out2 = dag.add_lut(b, FunctionTable::UNKWOWN, 3);
        dag.add_compositions([out1, out2], [input1, input2]);
        let analyzed_dag = super::analyze(&dag, &CONFIG, &None, PartitionIndex(1)).unwrap();
        assert_eq!(analyzed_dag.nb_partitions, 3);
        let actual_constraint_strings = analyzed_dag
            .variance_constraints
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let expected_constraint_strings = vec![
            "1σ²Br[0] + 1σ²Br[2] + 1σ²K[1→0] + 1σ²FK[0→1] + 1σ²FK[2→1] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] + 1σ²K[0→1] + 1σ²M[1] < (2²)**-10 (6bits partition:1 count:1, dom=20)",
            "1σ²Br[0] + 1σ²Br[1] + 1σ²Br[2] + 1σ²K[1→2] + 1σ²FK[0→1] + 1σ²FK[2→1] + 1σ²M[2] < (2²)**-17 (13bits partition:2 count:1, dom=34)",
            "1σ²Br[2] < (2²)**-7 (3bits partition:2 count:1, dom=14)",
            "1σ²Br[0] + 1σ²Br[1] + 1σ²Br[2] + 1σ²K[1→0] + 1σ²FK[0→1] + 1σ²FK[2→1] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
        ];
        assert_eq!(actual_constraint_strings, expected_constraint_strings);
        let partitions = [1, 1, 0, 1, 1, 1, 2, 0]
            .into_iter()
            .map(PartitionIndex)
            .collect::<Vec<_>>();
        assert_eq!(
            partitions,
            analyzed_dag
                .instrs_partition
                .iter()
                .map(|p| p.instruction_partition)
                .collect::<Vec<_>>()
        );
        assert!(analyzed_dag.instrs_partition[6]
            .alternative_output_representation
            .contains(&PartitionIndex(1)));
        assert!(analyzed_dag.instrs_partition[7]
            .alternative_output_representation
            .contains(&PartitionIndex(1)));
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_lut_sequence() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(8, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 8);
        let lut2 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 1);
        let lut3 = dag.add_lut(lut2, FunctionTable::UNKWOWN, 1);
        let lut4 = dag.add_lut(lut3, FunctionTable::UNKWOWN, 8);
        let lut5 = dag.add_lut(lut4, FunctionTable::UNKWOWN, 8);
        let partitions = [
            HIGH_PRECISION_PARTITION,
            HIGH_PRECISION_PARTITION,
            HIGH_PRECISION_PARTITION,
            LOW_PRECISION_PARTITION,
            LOW_PRECISION_PARTITION,
            HIGH_PRECISION_PARTITION,
        ];
        let dag = analyze(&dag);
        assert!(dag.nb_partitions == 2);
        for op_i in input1.0..=lut5.0 {
            let p = &dag.instrs_partition[op_i];
            let is_input = op_i == input1.0;
            assert!(p.instruction_partition == partitions[op_i]);
            if is_input {
                assert_input_on(&dag, p.instruction_partition, op_i, 1.0);
                assert_pbs_on(&dag, p.instruction_partition, op_i, 0.0);
            } else {
                assert_pbs_on(&dag, p.instruction_partition, op_i, 1.0);
                assert_input_on(&dag, p.instruction_partition, op_i, 0.0);
            }
        }
    }

    #[test]
    fn test_levelled_op() {
        let mut dag = unparametrized::Dag::new();
        let out_shape = Shape::number();
        let manp = 8.0;
        let input1 = dag.add_input(8, Shape::number());
        let input2 = dag.add_input(8, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 8);
        let _levelled = dag.add_linear_noise(
            [lut1, input2],
            LevelledComplexity::ZERO,
            [manp, manp],
            &out_shape,
            "comment",
        );
        let dag = analyze(&dag);
        assert!(dag.nb_partitions == 1);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_rounded_v3_first_layer_and_second_layer() {
        let acc_precision = 16;
        let precision = 8;
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(acc_precision, Shape::number());
        let rounded1 = dag.add_expanded_round(input1, precision);
        let lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let rounded2 = dag.add_expanded_round(lut1, precision);
        let lut2 = dag.add_lut(rounded2, FunctionTable::UNKWOWN, acc_precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        show_partitionning(&old_dag, &dag.instrs_partition);
        // First layer is fully LOW_PRECISION_PARTITION
        for op_i in input1.0..lut1.0 {
            let p = LOW_PRECISION_PARTITION;
            let sb = &dag.instrs_variances[op_i][p];
            assert!(sb.coeff(input_noise(p)) >= 1.0 || sb.coeff(bootstrap_noise(p)) >= 1.0);
        }
        // First lut is HIGH_PRECISION_PARTITION and immedialtely converted to LOW_PRECISION_PARTITION
        let p = HIGH_PRECISION_PARTITION;
        let sb = &dag.instrs_variances[lut1.0][p];
        assert!(sb.coeff(input_noise(p)) == 0.0);
        assert!(sb.coeff(bootstrap_noise(p)) == 1.0);
        let sb_after_fast_ks = &dag.instrs_variances[lut1.0][LOW_PRECISION_PARTITION];
        assert!(
            sb_after_fast_ks.coeff(fast_keyswitch_noise(
                HIGH_PRECISION_PARTITION,
                LOW_PRECISION_PARTITION
            )) == 1.0
        );
        // The next rounded is on LOW_PRECISION_PARTITION but base noise can comes from HIGH_PRECISION_PARTITION + FKS
        for op_i in (lut1.0 + 1)..lut2.0 {
            assert!(LOW_PRECISION_PARTITION == dag.instrs_partition[op_i].instruction_partition);
            let p = LOW_PRECISION_PARTITION;
            let sb = &dag.instrs_variances[op_i][p];
            // The base noise is either from the other partition and shifted or from the current partition and 1
            assert!(sb.coeff(input_noise(LOW_PRECISION_PARTITION)) == 0.0);
            assert!(sb.coeff(input_noise(HIGH_PRECISION_PARTITION)) == 0.0);
            if sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION)) >= 1.0 {
                assert!(
                    sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION))
                        == sb.coeff(fast_keyswitch_noise(
                            HIGH_PRECISION_PARTITION,
                            LOW_PRECISION_PARTITION
                        ))
                );
            } else {
                assert!(sb.coeff(bootstrap_noise(LOW_PRECISION_PARTITION)) == 1.0);
                assert!(
                    sb.coeff(fast_keyswitch_noise(
                        HIGH_PRECISION_PARTITION,
                        LOW_PRECISION_PARTITION
                    )) == 0.0
                );
            }
        }
        let sb = &dag.instrs_variances[lut2.0][HIGH_PRECISION_PARTITION];
        assert!(sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION)) >= 1.0);
    }

    #[allow(clippy::float_cmp, clippy::cognitive_complexity)]
    #[test]
    fn test_rounded_v3_classic_first_layer_second_layer() {
        let acc_precision = 16;
        let precision = 8;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let rounded1 = dag.add_expanded_round(input1, precision);
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        show_partitionning(&old_dag, &dag.instrs_partition);
        // First layer is fully HIGH_PRECISION_PARTITION
        assert!(
            dag.instrs_variances[free_input1.0][HIGH_PRECISION_PARTITION]
                .coeff(input_noise(HIGH_PRECISION_PARTITION))
                == 1.0
        );
        // First layer tlu
        let sb = &dag.instrs_variances[input1.0][HIGH_PRECISION_PARTITION];
        assert!(sb.coeff(input_noise(LOW_PRECISION_PARTITION)) == 0.0);
        assert!(sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION)) == 1.0);
        assert!(
            sb.coeff(fast_keyswitch_noise(
                HIGH_PRECISION_PARTITION,
                LOW_PRECISION_PARTITION
            )) == 0.0
        );
        // The same cyphertext exists in another partition with additional noise due to fast keyswitch
        let sb = &dag.instrs_variances[input1.0][LOW_PRECISION_PARTITION];
        assert!(sb.coeff(input_noise(LOW_PRECISION_PARTITION)) == 0.0);
        assert!(sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION)) == 1.0);
        assert!(
            sb.coeff(fast_keyswitch_noise(
                HIGH_PRECISION_PARTITION,
                LOW_PRECISION_PARTITION
            )) == 1.0
        );

        // Second layer
        let mut first_bit_extract_verified = false;
        let mut first_bit_erase_verified = false;
        for op_i in (input1.0 + 1)..rounded1.0 {
            if let Operator::Dot {
                weights, inputs, ..
            } = &dag.operators[op_i]
            {
                let bit_extract = weights.values.len() == 1;
                let first_bit_extract = bit_extract && !first_bit_extract_verified;
                let bit_erase = weights.values == [1, -1];
                let first_bit_erase = bit_erase && !first_bit_erase_verified;
                let input0_sb = &dag.instrs_variances[inputs[0].0][LOW_PRECISION_PARTITION];
                let input0_coeff_pbs_high =
                    input0_sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION));
                let input0_coeff_pbs_low =
                    input0_sb.coeff(bootstrap_noise(LOW_PRECISION_PARTITION));
                let input0_coeff_fks = input0_sb.coeff(fast_keyswitch_noise(
                    HIGH_PRECISION_PARTITION,
                    LOW_PRECISION_PARTITION,
                ));
                if bit_extract {
                    first_bit_extract_verified |= first_bit_extract;
                    assert!(input0_coeff_pbs_high >= 1.0);
                    if first_bit_extract {
                        assert!(input0_coeff_pbs_low == 0.0);
                    } else {
                        assert!(input0_coeff_pbs_low >= 1.0);
                    }
                    assert!(input0_coeff_fks == 1.0);
                } else if bit_erase {
                    first_bit_erase_verified |= first_bit_erase;
                    let input1_sb = &dag.instrs_variances[inputs[1].0][LOW_PRECISION_PARTITION];
                    let input1_coeff_pbs_high =
                        input1_sb.coeff(bootstrap_noise(HIGH_PRECISION_PARTITION));
                    let input1_coeff_pbs_low =
                        input1_sb.coeff(bootstrap_noise(LOW_PRECISION_PARTITION));
                    let input1_coeff_fks = input1_sb.coeff(fast_keyswitch_noise(
                        HIGH_PRECISION_PARTITION,
                        LOW_PRECISION_PARTITION,
                    ));
                    if first_bit_erase {
                        assert!(input0_coeff_pbs_low == 0.0);
                    } else {
                        assert!(input0_coeff_pbs_low >= 1.0);
                    }
                    assert!(input0_coeff_pbs_high == 1.0);
                    assert!(input0_coeff_fks == 1.0);
                    assert!(input1_coeff_pbs_low == 1.0);
                    assert!(input1_coeff_pbs_high == 0.0);
                    assert!(input1_coeff_fks == 0.0);
                }
            }
        }
        assert!(first_bit_extract_verified);
        assert!(first_bit_erase_verified);
    }

    #[test]
    fn test_rounded_v3_classic_first_layer_second_layer_constraints() {
        let acc_precision = 7;
        let precision = 4;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let rounded1 = dag.add_expanded_round(input1, precision);
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        show_partitionning(&old_dag, &dag.instrs_partition);
        let constraints: Vec<_> = dag
            .variance_constraints
            .iter()
            .map(VarianceConstraint::to_string)
            .collect();
        let expected_constraints = [
            // First lut to force partition HIGH_PRECISION_PARTITION
            "1σ²In[1] + 1σ²K[1] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=16)",
            // 16384(shift) = (2**7)², for Br[1]
            "16384σ²Br[1] + 1σ²K[0] + 16384σ²FK[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=22)",
            // 4096(shift) = (2**6)², 1(due to 1 erase bit) for Br[0] and 1 for Br[1]
            "4096σ²Br[0] + 4096σ²Br[1] + 1σ²K[0] + 4096σ²FK[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=20)",
            // 1024(shift) = (2**5)², 2(due to 2 erase bit for Br[0] and 1 for Br[1]
            "2048σ²Br[0] + 1024σ²Br[1] + 1σ²K[0] + 1024σ²FK[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=19)",
            // 3(erase bit) Br[0] and 1 initial Br[1]
            "3σ²Br[0] + 1σ²Br[1] + 1σ²K[0→1] + 1σ²FK[1→0] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=18)",
            // Last lut to close the cycle
            "1σ²Br[1] < (2²)**-8 (4bits partition:1 count:1, dom=16)",
        ];
        for (c, ec) in constraints.iter().zip(expected_constraints) {
            assert!(
                c == ec,
                "\nBad constraint\nActual: {c}\nTruth : {ec} (expected)\n"
            );
        }
        let simplified_constraints: Vec<_> = dag
            .undominated_variance_constraints
            .iter()
            .map(VarianceConstraint::to_string)
            .collect();
        let expected_simplified_constraints = [
            expected_constraints[1], // biggest weights on Br[1]
            expected_constraints[2], // biggest weights on Br[0]
            expected_constraints[4], // only one to have K[0→1]
            expected_constraints[0], // only one to have K[1]
                                     // 3 is dominated by 2
        ];
        for (c, ec) in simplified_constraints
            .iter()
            .zip(expected_simplified_constraints)
        {
            assert!(
                c == ec,
                "\nBad simplified constraint\nActual: {c}\nTruth : {ec} (expected)\n"
            );
        }
    }

    #[test]
    fn test_rounded_v1_classic_first_layer_second_layer_constraints() {
        let acc_precision = 7;
        let precision = 4;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        // let input1 = dag.add_input(acc_precision, Shape::number());
        let rounded1 = dag.add_expanded_round(input1, precision);
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, precision);
        let old_dag = dag;
        let dag = analyze_with_preferred(&old_dag, HIGH_PRECISION_PARTITION);
        show_partitionning(&old_dag, &dag.instrs_partition);
        let constraints: Vec<_> = dag
            .variance_constraints
            .iter()
            .map(VarianceConstraint::to_string)
            .collect();
        let expected_constraints = [
            // First lut to force partition HIGH_PRECISION_PARTITION
            "1σ²In[1] + 1σ²K[1] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=16)",
            // 16384(shift) = (2**7)², for Br[1]
            "16384σ²Br[1] + 1σ²K[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=22)",
            // 4096(shift) = (2**6)², 1(due to 1 erase bit) for Br[0] and 1 for Br[1]
            "4096σ²Br[0] + 4096σ²Br[1] + 1σ²K[1→0] + 4096σ²FK[0→1] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=20)",
            // 1024(shift) = (2**5)², 2(due to 2 erase bit for Br[0] and 1 for Br[1]
            "2048σ²Br[0] + 1024σ²Br[1] + 1σ²K[1→0] + 2048σ²FK[0→1] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=19)",
            "3σ²Br[0] + 1σ²Br[1] + 1σ²K[1] + 3σ²FK[0→1] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=18)",
        ];
        for (c, ec) in constraints.iter().zip(expected_constraints) {
            assert!(
                c == ec,
                "\nBad constraint\nActual: {c}\nTruth : {ec} (expected)\n"
            );
        }
        let simplified_constraints: Vec<_> = dag
            .undominated_variance_constraints
            .iter()
            .map(VarianceConstraint::to_string)
            .collect();
        let expected_simplified_constraints = [
            expected_constraints[1], // biggest weights on Br[1]
            expected_constraints[2], // biggest weights on Br[0]
            expected_constraints[4], // only one to have K[0→1]
            expected_constraints[0], // only one to have K[1]
                                     // 3 is dominated by 2
        ];
        for (c, ec) in simplified_constraints
            .iter()
            .zip(expected_simplified_constraints)
        {
            assert!(
                c == ec,
                "\nBad simplified constraint\nActual: {c}\nTruth : {ec} (expected)\n"
            );
        }
    }

    #[test]
    fn test_rounded_v3_classic_first_layer_second_layer_complexity() {
        let acc_precision = 7;
        let precision = 4;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let rounded1 = dag.add_expanded_round(input1, precision);
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        // Partition 0
        let instrs_counts: Vec<_> = dag
            .operations_count_per_instrs
            .iter()
            .map(OperationsCount::to_string)
            .collect();
        #[rustfmt::skip] // nighlty and stable are inconsitent here
        let expected_counts = [
            "∅",                            // free_input1
            "1¢Br[1] + 1¢K[1] + 1¢FK[1→0]", // input1
            "∅",                            // shift
            "∅",                            // cast
            "1¢Br[0] + 1¢K[0]",             // extract (lut)
            "∅",                            // erase (dot)
            "∅",                            // cast
            "∅",                            // shift
            "∅",                            // cast
            "1¢Br[0] + 1¢K[0]",             // extract (lut)
            "∅",                            // erase (dot)
            "∅",                            // cast
            "∅",                            // shift
            "∅",                            // cast
            "1¢Br[0] + 1¢K[0]",             // extract (lut)
            "∅",                            // erase (dot)
            "∅",                            // cast
            "1¢Br[1] + 1¢K[0→1]",           // _lut1
        ];
        for ((c, ec), op) in instrs_counts.iter().zip(expected_counts).zip(dag.operators) {
            assert!(
                c == ec,
                "\nBad count on {op}\nActual: {c}\nTruth : {ec} (expected)\n"
            );
        }
        eprintln!("{}", dag.operations_count);
        assert!(
            format!("{}", dag.operations_count)
                == "3¢Br[0] + 2¢Br[1] + 3¢K[0] + 1¢K[0→1] + 1¢K[1] + 1¢FK[1→0]"
        );
    }

    #[test]
    fn test_high_partition_number() {
        let mut dag = unparametrized::Dag::new();
        let max_precision = 10;
        let mut lut_input = dag.add_input(max_precision, Shape::number());
        for out_precision in (1..=max_precision).rev() {
            lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
        }
        _ = dag.add_lut(lut_input, FunctionTable::UNKWOWN, 1);
        let precisions: Vec<_> = (1..=max_precision).collect();
        let p_cut = PartitionCut::from_precisions(&precisions);
        let dag =
            super::analyze(&dag, &CONFIG, &Some(p_cut.clone()), LOW_PRECISION_PARTITION).unwrap();
        assert!(dag.nb_partitions == p_cut.n_partitions());
    }
}
