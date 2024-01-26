use crate::dag::operator::{
    dot_kind, DotKind, LevelledComplexity, Operator, OperatorIndex, Precision, Shape,
};
use crate::dag::rewrite::round::expand_round_and_index_map;
use crate::dag::unparametrized;
use crate::optimization::config::NoiseBoundConfig;
use crate::optimization::dag::multi_parameters::partition_cut::PartitionCut;
use crate::optimization::dag::multi_parameters::partitionning::partitionning_with_preferred;
use crate::optimization::dag::multi_parameters::partitions::{
    InstructionPartition, PartitionIndex, Transition,
};
use crate::optimization::dag::multi_parameters::symbolic_variance::SymbolicVariance;
use crate::optimization::dag::solo_key::analyze::{
    extra_final_values_to_check, first, safe_noise_bound,
};
use crate::optimization::Err::NotComposable;
use crate::optimization::Result;

use super::complexity::OperationsCount;
use super::keys_spec;
use super::operations_value::OperationsValue;
use super::variance_constraint::VarianceConstraint;

use crate::utils::square;

// private short convention
use DotKind as DK;

type Op = Operator;

#[derive(Debug)]
pub struct AnalyzedDag {
    pub operators: Vec<Op>,
    // Collect all operators ouput variances
    pub nb_partitions: usize,
    pub instrs_partition: Vec<InstructionPartition>,
    pub out_variances: Vec<Vec<SymbolicVariance>>,
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
    dag: &unparametrized::OperationDag,
    noise_config: &NoiseBoundConfig,
    p_cut: &Option<PartitionCut>,
    default_partition: PartitionIndex,
    composable: bool,
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
    let partitions = partitionning_with_preferred(&dag, &p_cut, default_partition, composable);
    let instrs_partition = partitions.instrs_partition;
    let nb_partitions = partitions.nb_partitions;
    let mut out_variances = self::out_variances(&dag, nb_partitions, &instrs_partition, &None);
    if composable {
        // Verify that there is no input symbol in the symbolic variances of the outputs.
        check_composability(&dag, &out_variances, nb_partitions)?;
        // Get the largest output out_variance
        let largest_output_variances = dag
            .get_output_index()
            .into_iter()
            .map(|index| out_variances[index].clone())
            .reduce(|lhs, rhs| {
                lhs.into_iter()
                    .zip(rhs)
                    .map(|(lhsi, rhsi)| lhsi.max(&rhsi))
                    .collect()
            })
            .expect("Failed to get the largest output variance.");
        // Re-compute the out variances with input variances overriden by input variances
        out_variances = self::out_variances(
            &dag,
            nb_partitions,
            &instrs_partition,
            &Some(largest_output_variances),
        );
    }
    let variance_constraints =
        collect_all_variance_constraints(&dag, noise_config, &instrs_partition, &out_variances);
    let undominated_variance_constraints =
        VarianceConstraint::remove_dominated(&variance_constraints);
    let operations_count_per_instrs =
        collect_operations_count(&dag, nb_partitions, &instrs_partition);
    let operations_count = sum_operations_count(&operations_count_per_instrs);
    Ok(AnalyzedDag {
        operators: dag.operators,
        instruction_rewrite_index,
        nb_partitions,
        instrs_partition,
        out_variances,
        levelled_complexity,
        variance_constraints,
        undominated_variance_constraints,
        operations_count_per_instrs,
        operations_count,
        p_cut,
    })
}

fn check_composability(
    dag: &unparametrized::OperationDag,
    symbolic_variances: &[Vec<SymbolicVariance>],
    nb_partitions: usize,
) -> Result<()> {
    // If the circuit only contains inputs, then it is composable.
    let only_inputs = dag
        .operators
        .iter()
        .all(|node| matches!(node, Operator::Input { .. }));
    if only_inputs {
        return Ok(());
    }

    // If the circuit outputs are free from input variances, it means that every outputs are
    // refreshed, and the function can be composed.
    let in_var_in_out_var = dag
        .get_output_index()
        .into_iter()
        .flat_map(|index| symbolic_variances[index].iter().map(move |v| (index, v)))
        .find_map(|(output_index, sym_var)| {
            (0..nb_partitions)
                .find(|partition| {
                    let coeff = sym_var.coeff_input(*partition);
                    coeff != 0.0f64 && !coeff.is_nan()
                })
                .map(|_| (output_index, sym_var))
        });
    match in_var_in_out_var {
        Some((output_id, sym_var)) => Err(NotComposable(format!(
            "Output {output_id} has variance {sym_var}."
        ))),
        None => Ok(()),
    }
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
            let new_instr_part = &dag.instrs_partition[new_instruction.i];
            if let Op::Lut { .. } = dag.operators[new_instruction.i] {
                let ks_dst = new_instr_part.instruction_partition;
                partition = Some(ks_dst);
                #[allow(clippy::match_on_vec_items)]
                let ks_src = match new_instr_part.inputs_transition[0] {
                    Some(Transition::Internal { src_partition }) => src_partition,
                    None => ks_dst,
                    _ => unreachable!(),
                };
                input_partition = Some(ks_src);
                let ks_key = ks_keys[ks_src][ks_dst].as_ref().unwrap().identifier;
                let pbs_key = pbs_keys[ks_dst].identifier;
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
                let key = fks_keys[src][dst].as_ref().unwrap().identifier;
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
            partition.unwrap_or(dag.instrs_partition[new_instructions[0].i].instruction_partition);
        let input_partition = input_partition.unwrap_or(partition);
        let merged = keys_spec::InstructionKeys {
            input_key: big_keys[input_partition].identifier,
            tlu_keyswitch_key: tlu_keyswitch_key.unwrap_or(unknown),
            tlu_bootstrap_key: tlu_bootstrap_key.unwrap_or(unknown),
            output_key: big_keys[partition].identifier,
            extra_conversion_keys: conversion_key.iter().copied().collect(),
            tlu_circuit_bootstrap_key: keys_spec::NO_KEY_ID,
            tlu_private_functional_packing_key: keys_spec::NO_KEY_ID,
        };
        result.push(merged);
    }
    result
}

fn out_variance(
    op: &unparametrized::UnparameterizedOperator,
    out_shapes: &[Shape],
    out_variances: &[Vec<SymbolicVariance>],
    nb_partitions: usize,
    instr_partition: &InstructionPartition,
    input_override: Option<Vec<SymbolicVariance>>,
) -> Vec<SymbolicVariance> {
    // If an override is given for input and we have an input node, we override.
    if let (Some(overr), Op::Input { .. }) = (input_override, op) {
        return overr;
    }
    // one variance per partition, in case the result is converted
    let partition = instr_partition.instruction_partition;
    let out_variance_of = |input: &OperatorIndex| {
        assert!(input.i < out_variances.len());
        assert!(partition < out_variances[input.i].len());
        assert!(out_variances[input.i][partition] != SymbolicVariance::ZERO);
        assert!(!out_variances[input.i][partition].coeffs.values[0].is_nan());
        assert!(out_variances[input.i][partition].partition != usize::MAX);
        out_variances[input.i][partition].clone()
    };
    let max_variance = |acc: SymbolicVariance, input: SymbolicVariance| acc.max(&input);
    let variance = match op {
        Op::Input { .. } => SymbolicVariance::input(nb_partitions, partition),
        Op::Lut { .. } => SymbolicVariance::after_pbs(nb_partitions, partition),
        Op::LevelledOp { inputs, manp, .. } => {
            let inputs_variance = inputs.iter().map(out_variance_of);
            let max_variance = inputs_variance.reduce(max_variance).unwrap();
            max_variance.after_levelled_op(*manp)
        }
        Op::Dot {
            inputs, weights, ..
        } => {
            let input_shape = first(inputs, out_shapes);
            let kind = dot_kind(inputs.len() as u64, input_shape, weights);
            match kind {
                DK::Simple | DK::Tensor | DK::Broadcast { .. } => {
                    let inputs_variance = (0..weights.values.len()).map(|j| {
                        let input = if inputs.len() > 1 {
                            inputs[j]
                        } else {
                            inputs[0]
                        };
                        out_variance_of(&input)
                    });
                    let mut out_variance = SymbolicVariance::ZERO;
                    for (input_variance, &weight) in inputs_variance.zip(&weights.values) {
                        assert!(input_variance != SymbolicVariance::ZERO);
                        out_variance += input_variance * square(weight as f64);
                    }
                    out_variance
                }
                DK::CompatibleTensor { .. } => todo!("TODO"),
                DK::Unsupported { .. } => panic!("Unsupported"),
            }
        }
        Op::UnsafeCast { input, .. } => out_variance_of(input),
        Op::Round { .. } => {
            unreachable!("Round should have been either expanded or integrated to a lut")
        }
    };
    // Injecting NAN in unused symbolic variance to detect bad use
    let unused = SymbolicVariance::nan(nb_partitions);
    let mut result = vec![unused; nb_partitions];
    for &dst_partition in &instr_partition.alternative_output_representation {
        let src_partition = partition;
        // make converted variance available in dst_partition
        result[dst_partition] =
            variance.after_partition_keyswitch_to_big(src_partition, dst_partition);
    }
    result[partition] = variance;
    result
}

fn out_variances(
    dag: &unparametrized::OperationDag,
    nb_partitions: usize,
    instrs_partition: &[InstructionPartition],
    input_override: &Option<Vec<SymbolicVariance>>,
) -> Vec<Vec<SymbolicVariance>> {
    let nb_ops = dag.operators.len();
    let mut out_variances = Vec::with_capacity(nb_ops);
    for (op, instr_partition) in dag.operators.iter().zip(instrs_partition) {
        let vf = out_variance(
            op,
            &dag.out_shapes,
            &out_variances,
            nb_partitions,
            instr_partition,
            input_override.clone(),
        );
        out_variances.push(vf);
    }
    out_variances
}

fn variance_constraint(
    dag: &unparametrized::OperationDag,
    noise_config: &NoiseBoundConfig,
    partition: PartitionIndex,
    op_i: usize,
    precision: Precision,
    variance: SymbolicVariance,
) -> VarianceConstraint {
    let nb_constraints = dag.out_shapes[op_i].flat_size();
    let safe_variance_bound = safe_noise_bound(precision, noise_config);
    VarianceConstraint {
        precision,
        partition,
        nb_constraints,
        safe_variance_bound,
        variance,
    }
}

#[allow(clippy::float_cmp)]
#[allow(clippy::match_on_vec_items)]
fn collect_all_variance_constraints(
    dag: &unparametrized::OperationDag,
    noise_config: &NoiseBoundConfig,
    instrs_partition: &[InstructionPartition],
    out_variances: &[Vec<SymbolicVariance>],
) -> Vec<VarianceConstraint> {
    let decryption_points = extra_final_values_to_check(dag);
    let mut constraints = vec![];
    for (op_i, op) in dag.operators.iter().enumerate() {
        let partition = instrs_partition[op_i].instruction_partition;
        if let Op::Lut { input, .. } = op {
            let precision = dag.out_precisions[input.i];
            let dst_partition = partition;
            let src_partition = match instrs_partition[op_i].inputs_transition[0] {
                None => dst_partition,
                Some(Transition::Internal { src_partition }) => {
                    assert!(src_partition != dst_partition);
                    src_partition
                }
                Some(Transition::Additional { src_partition }) => {
                    assert!(src_partition != dst_partition);
                    let variance = &out_variances[input.i][dst_partition];
                    assert!(
                        variance.coeff_partition_keyswitch_to_big(src_partition, dst_partition)
                            == 1.0
                    );
                    dst_partition
                }
            };
            let variance = &out_variances[input.i][src_partition].clone();
            let variance = variance
                .after_partition_keyswitch_to_small(src_partition, dst_partition)
                .after_modulus_switching(partition);
            constraints.push(variance_constraint(
                dag,
                noise_config,
                partition,
                op_i,
                precision,
                variance,
            ));
        }
        if decryption_points[op_i] {
            let precision = dag.out_precisions[op_i];
            let variance = out_variances[op_i][partition].clone();
            constraints.push(variance_constraint(
                dag,
                noise_config,
                partition,
                op_i,
                precision,
                variance,
            ));
        }
    }
    constraints
}

#[allow(clippy::match_on_vec_items)]
fn operations_counts(
    dag: &unparametrized::OperationDag,
    op: &unparametrized::UnparameterizedOperator,
    nb_partitions: usize,
    instr_partition: &InstructionPartition,
) -> OperationsCount {
    let mut counts = OperationsValue::zero(nb_partitions);
    if let Op::Lut { input, .. } = op {
        let partition = instr_partition.instruction_partition;
        let nb_lut = dag.out_shapes[input.i].flat_size() as f64;
        let src_partition = match instr_partition.inputs_transition[0] {
            Some(Transition::Internal { src_partition }) => src_partition,
            Some(Transition::Additional { .. }) | None => partition,
        };
        *counts.ks(src_partition, partition) += nb_lut;
        *counts.pbs(partition) += nb_lut;
        for &conv_partition in &instr_partition.alternative_output_representation {
            *counts.fks(partition, conv_partition) += nb_lut;
        }
    }
    OperationsCount { counts }
}

fn collect_operations_count(
    dag: &unparametrized::OperationDag,
    nb_partitions: usize,
    instrs_partition: &[InstructionPartition],
) -> Vec<OperationsCount> {
    dag.operators
        .iter()
        .enumerate()
        .map(|(i, op)| operations_counts(dag, op, nb_partitions, &instrs_partition[i]))
        .collect()
}

fn sum_operations_count(all_counts: &[OperationsCount]) -> OperationsCount {
    let mut sum_counts = OperationsValue::zero(all_counts[0].counts.nb_partitions());
    for OperationsCount { counts } in all_counts {
        sum_counts += counts;
    }
    OperationsCount { counts: sum_counts }
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

    pub fn analyze(dag: &unparametrized::OperationDag) -> AnalyzedDag {
        analyze_with_preferred(dag, LOW_PRECISION_PARTITION)
    }

    pub fn analyze_with_preferred(
        dag: &unparametrized::OperationDag,
        default_partition: PartitionIndex,
    ) -> AnalyzedDag {
        let p_cut = PartitionCut::for_each_precision(dag);
        super::analyze(dag, &CONFIG, &Some(p_cut), default_partition, false).unwrap()
    }

    #[allow(clippy::float_cmp)]
    fn assert_input_on(dag: &AnalyzedDag, partition: usize, op_i: usize, expected_coeff: f64) {
        for symbolic_variance_partition in [LOW_PRECISION_PARTITION, HIGH_PRECISION_PARTITION] {
            let sb = dag.out_variances[op_i][partition].clone();
            let coeff = if sb == SymbolicVariance::ZERO {
                0.0
            } else {
                sb.coeff_input(symbolic_variance_partition)
            };
            if symbolic_variance_partition == partition {
                assert!(
                    coeff == expected_coeff,
                    "INCORRECT INPUT COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.out_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            } else {
                assert!(
                    coeff == 0.0,
                    "INCORRECT INPUT COEFF ON WRONG PARTITION {:?} {:?} {} {}",
                    dag.out_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            }
        }
    }

    #[allow(clippy::float_cmp)]
    fn assert_pbs_on(dag: &AnalyzedDag, partition: usize, op_i: usize, expected_coeff: f64) {
        for symbolic_variance_partition in [LOW_PRECISION_PARTITION, HIGH_PRECISION_PARTITION] {
            let sb = dag.out_variances[op_i][partition].clone();
            eprintln!("{:?}", dag.out_variances[op_i]);
            eprintln!("{:?}", dag.out_variances[op_i][partition]);
            let coeff = if sb == SymbolicVariance::ZERO {
                0.0
            } else {
                sb.coeff_pbs(symbolic_variance_partition)
            };
            if symbolic_variance_partition == partition {
                assert!(
                    coeff == expected_coeff,
                    "INCORRECT PBS COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.out_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            } else {
                assert!(
                    coeff == 0.0,
                    "INCORRECT PBS COEFF ON GOOD PARTITION {:?} {:?} {} {}",
                    dag.out_variances[op_i],
                    partition,
                    coeff,
                    expected_coeff
                );
            }
        }
    }

    #[test]
    fn test_composition_with_inputs_only() {
        let mut dag = unparametrized::OperationDag::new();
        let _ = dag.add_input(1, Shape::number());
        let p_cut = PartitionCut::for_each_precision(&dag);
        let res = super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION, true);
        assert!(res.is_ok());
    }

    #[test]
    fn test_composition_1_partition() {
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(1, Shape::number());
        let _ = dag.add_lut(input1, FunctionTable::UNKWOWN, 2);
        let p_cut = PartitionCut::for_each_precision(&dag);
        let dag =
            super::analyze(&dag, &CONFIG, &Some(p_cut), LOW_PRECISION_PARTITION, true).unwrap();
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
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(3, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 6);
        let lut3 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 3);
        let input2 = dag.add_dot([input1, lut3], [1, 1]);
        let _ = dag.add_lut(input2, FunctionTable::UNKWOWN, 3);
        let analyzed_dag =
            super::analyze(&dag, &CONFIG, &None, LOW_PRECISION_PARTITION, true).unwrap();
        assert_eq!(analyzed_dag.nb_partitions, 2);
        let actual_constraint_strings = analyzed_dag
            .variance_constraints
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let expected_constraint_strings = vec![
            "1σ²Br[0] + 1σ²K[0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] + 1σ²K[0→1] + 1σ²M[1] < (2²)**-10 (6bits partition:1 count:1, dom=20)",
            "1σ²Br[0] + 1σ²Br[1] + 1σ²FK[1→0] + 1σ²K[0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
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
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(3, Shape::number());
        let input2 = dag.add_input(13, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 6);
        let lut3 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 3);
        let a = dag.add_dot([input2, lut3], [1, 1]);
        let b = dag.add_dot([input1, lut3], [1, 1]);
        let _ = dag.add_lut(a, FunctionTable::UNKWOWN, 3);
        let _ = dag.add_lut(b, FunctionTable::UNKWOWN, 3);
        let analyzed_dag = super::analyze(&dag, &CONFIG, &None, 1, true).unwrap();
        assert_eq!(analyzed_dag.nb_partitions, 3);
        let actual_constraint_strings = analyzed_dag
            .variance_constraints
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<String>>();
        let expected_constraint_strings = vec![
            "1σ²Br[0] + 1σ²FK[0→1] + 1σ²Br[2] + 1σ²FK[2→1] + 1σ²K[1→0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] + 1σ²K[0→1] + 1σ²M[1] < (2²)**-10 (6bits partition:1 count:1, dom=20)",
            "1σ²Br[0] + 1σ²FK[0→1] + 1σ²Br[1] + 1σ²Br[2] + 1σ²FK[2→1] + 1σ²K[1→2] + 1σ²M[2] < (2²)**-17 (13bits partition:2 count:1, dom=34)",
            "1σ²Br[2] < (2²)**-7 (3bits partition:2 count:1, dom=14)",
            "1σ²Br[0] + 1σ²FK[0→1] + 1σ²Br[1] + 1σ²Br[2] + 1σ²FK[2→1] + 1σ²K[1→0] + 1σ²M[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
            "1σ²Br[0] < (2²)**-7 (3bits partition:0 count:1, dom=14)",
        ];
        assert_eq!(actual_constraint_strings, expected_constraint_strings);
        let partitions = vec![1, 1, 0, 1, 1, 1, 2, 0];
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
            .contains(&1));
        assert!(analyzed_dag.instrs_partition[7]
            .alternative_output_representation
            .contains(&1));
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_lut_sequence() {
        let mut dag = unparametrized::OperationDag::new();
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
        for op_i in input1.i..=lut5.i {
            let p = &dag.instrs_partition[op_i];
            let is_input = op_i == input1.i;
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
        let mut dag = unparametrized::OperationDag::new();
        let out_shape = Shape::number();
        let manp = 8.0;
        let input1 = dag.add_input(8, Shape::number());
        let input2 = dag.add_input(8, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 8);
        let _levelled = dag.add_levelled_op(
            [lut1, input2],
            LevelledComplexity::ZERO,
            manp,
            &out_shape,
            "comment",
        );
        let dag = analyze(&dag);
        assert!(dag.nb_partitions == 1);
    }

    fn nan_symbolic_variance(sb: &SymbolicVariance) -> bool {
        sb.coeffs[0].is_nan()
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_rounded_v3_first_layer_and_second_layer() {
        let acc_precision = 16;
        let precision = 8;
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(acc_precision, Shape::number());
        let rounded1 = dag.add_expanded_round(input1, precision);
        let lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let rounded2 = dag.add_expanded_round(lut1, precision);
        let lut2 = dag.add_lut(rounded2, FunctionTable::UNKWOWN, acc_precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        show_partitionning(&old_dag, &dag.instrs_partition);
        // First layer is fully LOW_PRECISION_PARTITION
        for op_i in input1.i..lut1.i {
            let p = LOW_PRECISION_PARTITION;
            let sb = &dag.out_variances[op_i][p];
            assert!(sb.coeff_input(p) >= 1.0 || sb.coeff_pbs(p) >= 1.0);
            assert!(nan_symbolic_variance(
                &dag.out_variances[op_i][HIGH_PRECISION_PARTITION]
            ));
        }
        // First lut is HIGH_PRECISION_PARTITION and immedialtely converted to LOW_PRECISION_PARTITION
        let p = HIGH_PRECISION_PARTITION;
        let sb = &dag.out_variances[lut1.i][p];
        assert!(sb.coeff_input(p) == 0.0);
        assert!(sb.coeff_pbs(p) == 1.0);
        let sb_after_fast_ks = &dag.out_variances[lut1.i][LOW_PRECISION_PARTITION];
        assert!(
            sb_after_fast_ks.coeff_partition_keyswitch_to_big(
                HIGH_PRECISION_PARTITION,
                LOW_PRECISION_PARTITION
            ) == 1.0
        );
        // The next rounded is on LOW_PRECISION_PARTITION but base noise can comes from HIGH_PRECISION_PARTITION + FKS
        for op_i in (lut1.i + 1)..lut2.i {
            assert!(LOW_PRECISION_PARTITION == dag.instrs_partition[op_i].instruction_partition);
            let p = LOW_PRECISION_PARTITION;
            let sb = &dag.out_variances[op_i][p];
            // The base noise is either from the other partition and shifted or from the current partition and 1
            assert!(sb.coeff_input(LOW_PRECISION_PARTITION) == 0.0);
            assert!(sb.coeff_input(HIGH_PRECISION_PARTITION) == 0.0);
            if sb.coeff_pbs(HIGH_PRECISION_PARTITION) >= 1.0 {
                assert!(
                    sb.coeff_pbs(HIGH_PRECISION_PARTITION)
                        == sb.coeff_partition_keyswitch_to_big(
                            HIGH_PRECISION_PARTITION,
                            LOW_PRECISION_PARTITION
                        )
                );
            } else {
                assert!(sb.coeff_pbs(LOW_PRECISION_PARTITION) == 1.0);
                assert!(
                    sb.coeff_partition_keyswitch_to_big(
                        HIGH_PRECISION_PARTITION,
                        LOW_PRECISION_PARTITION
                    ) == 0.0
                );
            }
        }
        assert!(nan_symbolic_variance(
            &dag.out_variances[lut2.i][LOW_PRECISION_PARTITION]
        ));
        let sb = &dag.out_variances[lut2.i][HIGH_PRECISION_PARTITION];
        assert!(sb.coeff_pbs(HIGH_PRECISION_PARTITION) >= 1.0);
    }

    #[allow(clippy::float_cmp, clippy::cognitive_complexity)]
    #[test]
    fn test_rounded_v3_classic_first_layer_second_layer() {
        let acc_precision = 16;
        let precision = 8;
        let mut dag = unparametrized::OperationDag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let rounded1 = dag.add_expanded_round(input1, precision);
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let old_dag = dag;
        let dag = analyze(&old_dag);
        show_partitionning(&old_dag, &dag.instrs_partition);
        // First layer is fully HIGH_PRECISION_PARTITION
        assert!(
            dag.out_variances[free_input1.i][HIGH_PRECISION_PARTITION]
                .coeff_input(HIGH_PRECISION_PARTITION)
                == 1.0
        );
        // First layer tlu
        let sb = &dag.out_variances[input1.i][HIGH_PRECISION_PARTITION];
        assert!(sb.coeff_input(LOW_PRECISION_PARTITION) == 0.0);
        assert!(sb.coeff_pbs(HIGH_PRECISION_PARTITION) == 1.0);
        assert!(
            sb.coeff_partition_keyswitch_to_big(HIGH_PRECISION_PARTITION, LOW_PRECISION_PARTITION)
                == 0.0
        );
        // The same cyphertext exists in another partition with additional noise due to fast keyswitch
        let sb = &dag.out_variances[input1.i][LOW_PRECISION_PARTITION];
        assert!(sb.coeff_input(LOW_PRECISION_PARTITION) == 0.0);
        assert!(sb.coeff_pbs(HIGH_PRECISION_PARTITION) == 1.0);
        assert!(
            sb.coeff_partition_keyswitch_to_big(HIGH_PRECISION_PARTITION, LOW_PRECISION_PARTITION)
                == 1.0
        );

        // Second layer
        let mut first_bit_extract_verified = false;
        let mut first_bit_erase_verified = false;
        for op_i in (input1.i + 1)..rounded1.i {
            if let Op::Dot {
                weights, inputs, ..
            } = &dag.operators[op_i]
            {
                let bit_extract = weights.values.len() == 1;
                let first_bit_extract = bit_extract && !first_bit_extract_verified;
                let bit_erase = weights.values == [1, -1];
                let first_bit_erase = bit_erase && !first_bit_erase_verified;
                let input0_sb = &dag.out_variances[inputs[0].i][LOW_PRECISION_PARTITION];
                let input0_coeff_pbs_high = input0_sb.coeff_pbs(HIGH_PRECISION_PARTITION);
                let input0_coeff_pbs_low = input0_sb.coeff_pbs(LOW_PRECISION_PARTITION);
                let input0_coeff_fks = input0_sb.coeff_partition_keyswitch_to_big(
                    HIGH_PRECISION_PARTITION,
                    LOW_PRECISION_PARTITION,
                );
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
                    let input1_sb = &dag.out_variances[inputs[1].i][LOW_PRECISION_PARTITION];
                    let input1_coeff_pbs_high = input1_sb.coeff_pbs(HIGH_PRECISION_PARTITION);
                    let input1_coeff_pbs_low = input1_sb.coeff_pbs(LOW_PRECISION_PARTITION);
                    let input1_coeff_fks = input1_sb.coeff_partition_keyswitch_to_big(
                        HIGH_PRECISION_PARTITION,
                        LOW_PRECISION_PARTITION,
                    );
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
        let mut dag = unparametrized::OperationDag::new();
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
            "16384σ²Br[1] + 16384σ²FK[1→0] + 1σ²K[0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=22)",
            // 4096(shift) = (2**6)², 1(due to 1 erase bit) for Br[0] and 1 for Br[1]
            "4096σ²Br[0] + 4096σ²Br[1] + 4096σ²FK[1→0] + 1σ²K[0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=20)",
            // 1024(shift) = (2**5)², 2(due to 2 erase bit for Br[0] and 1 for Br[1]
            "2048σ²Br[0] + 1024σ²Br[1] + 1024σ²FK[1→0] + 1σ²K[0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=19)",
            // 3(erase bit) Br[0] and 1 initial Br[1]
            "3σ²Br[0] + 1σ²Br[1] + 1σ²FK[1→0] + 1σ²K[0→1] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=18)",
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
        let mut dag = unparametrized::OperationDag::new();
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
            "4096σ²Br[0] + 4096σ²FK[0→1] + 4096σ²Br[1] + 1σ²K[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=20)",
            // 1024(shift) = (2**5)², 2(due to 2 erase bit for Br[0] and 1 for Br[1]
            "2048σ²Br[0] + 2048σ²FK[0→1] + 1024σ²Br[1] + 1σ²K[1→0] + 1σ²M[0] < (2²)**-4 (0bits partition:0 count:1, dom=19)",
            "3σ²Br[0] + 3σ²FK[0→1] + 1σ²Br[1] + 1σ²K[1] + 1σ²M[1] < (2²)**-8 (4bits partition:1 count:1, dom=18)",
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
        let mut dag = unparametrized::OperationDag::new();
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
            "ZERO x ¢",                     // free_input1
            "1¢K[1] + 1¢Br[1] + 1¢FK[1→0]", // input1
            "ZERO x ¢",                     // shift
            "ZERO x ¢",                     // cast
            "1¢K[0] + 1¢Br[0]",             // extract (lut)
            "ZERO x ¢",                     // erase (dot)
            "ZERO x ¢",                     // cast
            "ZERO x ¢",                     // shift
            "ZERO x ¢",                     // cast
            "1¢K[0] + 1¢Br[0]",             // extract (lut)
            "ZERO x ¢",                     // erase (dot)
            "ZERO x ¢",                     // cast
            "ZERO x ¢",                     // shift
            "ZERO x ¢",                     // cast
            "1¢K[0] + 1¢Br[0]",             // extract (lut)
            "ZERO x ¢",                     // erase (dot)
            "ZERO x ¢",                     // cast
            "1¢K[0→1] + 1¢Br[1]",           // _lut1
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
                == "3¢K[0] + 1¢K[0→1] + 1¢K[1] + 3¢Br[0] + 2¢Br[1] + 1¢FK[1→0]"
        );
    }

    #[test]
    fn test_high_partition_number() {
        let mut dag = unparametrized::OperationDag::new();
        let max_precision = 10;
        let mut lut_input = dag.add_input(max_precision, Shape::number());
        for out_precision in (1..=max_precision).rev() {
            lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
        }
        _ = dag.add_lut(lut_input, FunctionTable::UNKWOWN, 1);
        let precisions: Vec<_> = (1..=max_precision).collect();
        let p_cut = PartitionCut::from_precisions(&precisions);
        let dag = super::analyze(
            &dag,
            &CONFIG,
            &Some(p_cut.clone()),
            LOW_PRECISION_PARTITION,
            false,
        )
        .unwrap();
        assert!(dag.nb_partitions == p_cut.p_cut.len() + 1);
    }
}
