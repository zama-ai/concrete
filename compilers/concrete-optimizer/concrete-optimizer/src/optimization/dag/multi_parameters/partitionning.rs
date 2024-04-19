use std::collections::{HashMap, HashSet};

use crate::dag::operator::{Operator, OperatorIndex};
use crate::dag::unparametrized;

use super::partition_cut::PartitionCut;
use super::partitions::{InstructionPartition, PartitionIndex, Partitions, Transition};
use super::union_find::UnionFind;
type Op = Operator;

// Blocks of instructions
pub struct Blocks {
    // Set of instructions indexes for each block
    pub blocks: Vec<Vec<usize>>,
    // Block index of each instructions
    pub block_of: Vec<usize>,
}

impl Blocks {
    pub fn from(mut uf: UnionFind) -> Self {
        let mut block_of_canon: HashMap<usize, usize> = HashMap::new();
        let mut blocks: Vec<Vec<usize>> = vec![];
        let size = uf.parent.len();
        for op_i in 0..size {
            let canon = uf.find_canonical(op_i);
            // the canonic is always the smaller, so it's the first
            if canon == op_i {
                let block_i = blocks.len();
                _ = block_of_canon.insert(canon, block_i);
                blocks.push(vec![canon]);
            } else {
                let &block_i = block_of_canon.get(&canon).unwrap();
                blocks[block_i].push(op_i);
            }
        }
        let mut block_of = vec![0; size];
        for (i, block) in blocks.iter().enumerate() {
            for &a in block {
                block_of[a] = i;
            }
        }
        Self { blocks, block_of }
    }
}

// Extract block of instructions connected by levelled ops.
// This facilitates reasonning about conflicts on levelled ops.
#[allow(clippy::match_same_arms)]
fn extract_levelled_block(dag: &unparametrized::Dag) -> Blocks {
    let mut uf = UnionFind::new(dag.operators.len());
    for (op_i, op) in dag.operators.iter().enumerate() {
        match op {
            // Block entry point
            Operator::Input { .. } => (),
            // Block entry point and pre-exit point
            Op::Lut { .. } => (),
            // Connectors
            Op::UnsafeCast { input, .. } => uf.union(input.0, op_i),
            Op::LevelledOp { inputs, .. } | Op::Dot { inputs, .. } => {
                for input in inputs {
                    uf.union(input.0, op_i);
                }
            }
            Op::Round { .. } => unreachable!("Round should have been expanded"),
        };
    }
    // We apply the composition rules
    for (to_id, froms) in dag.composition.clone() {
        for from_id in froms {
            uf.union(to_id.0, from_id.0);
        }
    }
    Blocks::from(uf)
}

#[derive(Clone, Debug, Default)]
struct BlockConstraints {
    forced: HashSet<PartitionIndex>, // hard constraints, need to be resolved, given by PartitionFromOp
    exit: HashSet<PartitionIndex>, // soft constraints, to have less inter partition keyswitch in TLUs
}

/* For each levelled block collect BlockConstraints */
fn levelled_blocks_constraints(
    dag: &unparametrized::Dag,
    blocks: &Blocks,
    p_cut: &PartitionCut,
) -> Vec<BlockConstraints> {
    let mut constraints_by_block = vec![BlockConstraints::default(); blocks.blocks.len()];
    for (block_i, ops_i) in blocks.blocks.iter().enumerate() {
        for &op_i in ops_i {
            let op = &dag.operators[op_i];
            if let Some(partition) = p_cut.partition(dag, OperatorIndex(op_i)) {
                _ = constraints_by_block[block_i].forced.insert(partition);
                if let Some(input) = op_tlu_inputs(op) {
                    let input_group = blocks.block_of[input.0];
                    constraints_by_block[input_group].exit.extend([partition]);
                }
            }
        }
    }
    constraints_by_block
}

fn op_tlu_inputs(op: &Operator) -> Option<OperatorIndex> {
    match op {
        Op::Lut { input, .. } => Some(*input),
        _ => None,
    }
}

fn get_singleton_value<V: Copy>(hashset: &HashSet<V>) -> V {
    *hashset.iter().next().unwrap()
}

fn only_1_partition(dag: &unparametrized::Dag) -> Partitions {
    let mut instrs_partition =
        vec![InstructionPartition::new(PartitionIndex::FIRST); dag.operators.len()];
    for (op_i, op) in dag.operators.iter().enumerate() {
        match op {
            Op::Dot { inputs, .. } | Op::LevelledOp { inputs, .. } => {
                instrs_partition[op_i].inputs_transition = vec![None; inputs.len()];
            }
            Op::Lut { .. } | Op::UnsafeCast { .. } => {
                instrs_partition[op_i].inputs_transition = vec![None];
            }
            Op::Input { .. } => (),
            Op::Round { .. } => unreachable!(),
        }
    }
    Partitions {
        nb_partitions: 1,
        instrs_partition,
    }
}

fn resolve_by_levelled_block(
    dag: &unparametrized::Dag,
    p_cut: &PartitionCut,
    default_partition: PartitionIndex,
) -> Partitions {
    let blocks = extract_levelled_block(dag);
    let constraints_by_blocks = levelled_blocks_constraints(dag, &blocks, p_cut);
    let present_partitions: HashSet<PartitionIndex> = constraints_by_blocks
        .iter()
        .flat_map(|c| &c.forced)
        .copied()
        .collect();
    let nb_partitions = present_partitions.len().max(1); // no tlu = no constraints
    if p_cut.p_cut.len() + 1 != nb_partitions {
        return resolve_by_levelled_block(
            dag,
            &p_cut.delete_unused_cut(&present_partitions),
            default_partition,
        );
    }
    if nb_partitions == 1 {
        return only_1_partition(dag);
    }
    let mut block_partition: Vec<PartitionIndex> = vec![];
    for constraints in constraints_by_blocks {
        let partition = match constraints.forced.len() {
            0 => {
                if constraints.exit.len() == 1 {
                    get_singleton_value(&constraints.exit)
                } else {
                    default_partition
                }
            }
            1 => get_singleton_value(&constraints.forced),
            _ => {
                let forced = constraints.forced;
                if forced.contains(&default_partition) {
                    default_partition
                } else {
                    *forced.iter().min().unwrap()
                }
            }
        };
        // TODO1: Could choose based on the number of fast keyswitch added (case > 1)
        // TODO2: A conversion of an entry point could be deffered to the conflict until a conversion is needed
        //        This is equivalent to refine levelled block
        // TODO3: This could make even make some exit value used in a different representation and go out unconverted
        //        This can reduce the need to define extra parameters for internal ks
        block_partition.push(partition);
    }
    let mut instrs_p: Vec<InstructionPartition> =
        vec![InstructionPartition::new(default_partition); dag.operators.len()];
    let block_partition_of = |op_i| block_partition[blocks.block_of[op_i]];
    for (op_i, op) in dag.operators.iter().enumerate() {
        let group_partition = block_partition_of(op_i);
        match op {
            Op::Lut { input, .. } => {
                let instruction_partition = p_cut.partition(dag, OperatorIndex(op_i)).unwrap();
                instrs_p[op_i].instruction_partition = instruction_partition;
                let input_partition = instrs_p[input.0].instruction_partition;
                instrs_p[op_i].inputs_transition = if input_partition == instruction_partition {
                    vec![None]
                } else {
                    vec![Some(Transition::Internal {
                        src_partition: input_partition,
                    })]
                };
                if group_partition != instruction_partition {
                    instrs_p[op_i].alternative_output_representation =
                        HashSet::from([group_partition]);
                }
            }
            Op::LevelledOp { inputs, .. } | Op::Dot { inputs, .. } => {
                instrs_p[op_i].instruction_partition = group_partition;
                instrs_p[op_i].inputs_transition = vec![None; inputs.len()];
                for (i, input) in inputs.iter().enumerate() {
                    let input_partition = instrs_p[input.0].instruction_partition;
                    if group_partition != input_partition {
                        instrs_p[op_i].inputs_transition[i] = Some(Transition::Additional {
                            src_partition: input_partition,
                        });
                    }
                }
            }
            Op::UnsafeCast { input, .. } => {
                instrs_p[op_i].instruction_partition = group_partition;
                let input_partition = instrs_p[input.0].instruction_partition;
                instrs_p[op_i].inputs_transition = if group_partition == input_partition {
                    vec![None]
                } else {
                    vec![Some(Transition::Additional {
                        src_partition: input_partition,
                    })]
                }
            }
            Operator::Input { .. } => instrs_p[op_i].instruction_partition = group_partition,
            Op::Round { .. } => unreachable!("Round should have been expanded"),
        }
    }
    Partitions {
        nb_partitions,
        instrs_partition: instrs_p,
    }
    // Now we can generate transitions
    // Input has no transtions
    // Tlu has internal transtions based on input partition
    // Tlu has immediate external transition if needed
}

pub fn partitionning_with_preferred(
    dag: &unparametrized::Dag,
    p_cut: &PartitionCut,
    default_partition: PartitionIndex,
) -> Partitions {
    if p_cut.p_cut.is_empty() {
        only_1_partition(dag)
    } else {
        resolve_by_levelled_block(dag, p_cut, default_partition)
    }
}

#[cfg(test)]
pub mod tests {

    // 2 Partitions labels
    pub const LOW_PRECISION_PARTITION: PartitionIndex = PartitionIndex(0);
    pub const HIGH_PRECISION_PARTITION: PartitionIndex = PartitionIndex(1);

    use super::*;
    use crate::dag::operator::{FunctionTable, Shape, Weights};
    use crate::dag::unparametrized;

    fn default_p_cut() -> PartitionCut {
        PartitionCut::from_precisions(&[2, 128])
    }

    fn partitionning_no_p_cut(dag: &unparametrized::Dag) -> Partitions {
        let p_cut = PartitionCut::empty();
        partitionning_with_preferred(dag, &p_cut, LOW_PRECISION_PARTITION)
    }

    fn partitionning(dag: &unparametrized::Dag) -> Partitions {
        partitionning_with_preferred(
            dag,
            &PartitionCut::for_each_precision(dag),
            LOW_PRECISION_PARTITION,
        )
    }

    fn partitionning_with_preferred(
        dag: &unparametrized::Dag,
        p_cut: &PartitionCut,
        default_partition: PartitionIndex,
    ) -> Partitions {
        super::partitionning_with_preferred(dag, p_cut, default_partition)
    }

    pub fn show_partitionning(dag: &unparametrized::Dag, partitions: &[InstructionPartition]) {
        println!("Dag:");
        for (i, op) in dag.operators.iter().enumerate() {
            let partition = partitions[i].instruction_partition;
            print!("P {partition}");
            if partitions[i].alternative_output_representation.is_empty() {
                print!(" _");
            } else {
                print!(" +FKS{:?}", partitions[i].alternative_output_representation);
            };
            // partition
            if !partitions[i].inputs_transition.is_empty() {
                print!(" <- ");
                // type in
                for arg in &partitions[i].inputs_transition {
                    match arg {
                        None => print!("_,"),
                        Some(Transition::Internal { src_partition }) => {
                            print!("{src_partition}&KS,");
                        }
                        Some(Transition::Additional { src_partition }) => {
                            print!("{src_partition}+FKS,");
                        }
                    };
                }
            }
            println!();
            println!("%{i} <- {op}");
        }
    }

    #[test]
    fn test_1_partition() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(16, Shape::number());
        _ = dag.add_expanded_rounded_lut(input1, FunctionTable::UNKWOWN, 4, 8);
        let instrs_partition = partitionning_no_p_cut(&dag).instrs_partition;
        for instr_partition in instrs_partition {
            assert!(instr_partition.instruction_partition == LOW_PRECISION_PARTITION);
            assert!(instr_partition.no_transition());
        }
    }

    #[test]
    fn test_1_input_2_partitions() {
        let mut dag = unparametrized::Dag::new();
        _ = dag.add_input(1, Shape::number());
        let partitions = partitionning(&dag);
        assert!(partitions.nb_partitions == 1);
        let instrs_partition = partitions.instrs_partition;
        assert!(instrs_partition[0].instruction_partition == LOW_PRECISION_PARTITION);
        assert!(partitions.nb_partitions == 1);
    }

    #[test]
    fn test_2_partitions_with_without_compo() {
        let mut dag = unparametrized::Dag::new();
        let input = dag.add_input(10, Shape::number());
        let lut1 = dag.add_lut(input, FunctionTable::UNKWOWN, 2);
        let output = dag.add_lut(lut1, FunctionTable::UNKWOWN, 10);
        let partitions = partitionning(&dag);
        assert!(
            partitions.instrs_partition[input.0].instruction_partition
                != partitions.instrs_partition[output.0].instruction_partition
        );
        dag.add_composition(output, input);
        let partitions = partitionning(&dag);
        assert!(
            partitions.instrs_partition[input.0].instruction_partition
                == partitions.instrs_partition[output.0].instruction_partition
        );
    }

    #[test]
    fn test_2_lut_sequence() {
        let mut dag = unparametrized::Dag::new();
        let mut expected_partitions = vec![];
        let input1 = dag.add_input(8, Shape::number());
        expected_partitions.push(HIGH_PRECISION_PARTITION);
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 8);
        expected_partitions.push(HIGH_PRECISION_PARTITION);
        let lut2 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 1);
        expected_partitions.push(HIGH_PRECISION_PARTITION);
        let lut3 = dag.add_lut(lut2, FunctionTable::UNKWOWN, 1);
        expected_partitions.push(LOW_PRECISION_PARTITION);
        let lut4 = dag.add_lut(lut3, FunctionTable::UNKWOWN, 8);
        expected_partitions.push(LOW_PRECISION_PARTITION);
        let lut5 = dag.add_lut(lut4, FunctionTable::UNKWOWN, 8);
        expected_partitions.push(HIGH_PRECISION_PARTITION);
        let partitions = partitionning(&dag);
        assert!(partitions.nb_partitions == 2);
        let instrs_partition = partitions.instrs_partition;
        let consider = |op_i: OperatorIndex| &instrs_partition[op_i.0];
        show_partitionning(&dag, &instrs_partition);
        assert!(consider(input1).instruction_partition == HIGH_PRECISION_PARTITION); // no constraint
        assert!(consider(lut1).instruction_partition == expected_partitions[1]);
        assert!(consider(lut2).instruction_partition == expected_partitions[2]);
        assert!(consider(lut3).instruction_partition == expected_partitions[3]);
        assert!(consider(lut4).instruction_partition == expected_partitions[4]);
        assert!(consider(lut5).instruction_partition == expected_partitions[5]);
        assert!(instrs_partition.len() == 6);
    }

    #[test]
    fn test_mixed_dot_no_conflict_low() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(8, Shape::number());
        let input2 = dag.add_input(1, Shape::number());
        let lut2 = dag.add_lut(input2, FunctionTable::UNKWOWN, 8);
        let _dot = dag.add_dot([input1, lut2], Weights::from([1, 1]));
        let partitions = partitionning(&dag);
        assert!(partitions.nb_partitions == 1);
    }

    #[test]
    fn test_mixed_dot_no_conflict_high() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(8, Shape::number());
        let input2 = dag.add_input(1, Shape::number());
        let lut2 = dag.add_lut(input1, FunctionTable::UNKWOWN, 1);
        let _dot = dag.add_dot([input2, lut2], Weights::from([1, 1]));
        let partitions = partitionning(&dag);
        assert!(partitions.nb_partitions == 1);
    }

    #[test]
    fn test_mixed_dot_conflict() {
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(8, Shape::number());
        let input2 = dag.add_input(1, Shape::number());
        let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 8);
        let lut2 = dag.add_lut(input2, FunctionTable::UNKWOWN, 8);
        let dot = dag.add_dot([lut1, lut2], Weights::from([1, 1]));
        let partitions = partitionning(&dag);
        let consider = |op_i: OperatorIndex| &partitions.instrs_partition[op_i.0];
        // input1
        let p = consider(input1);
        {
            assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
            assert!(p.no_transition());
        };
        // input2
        let p = consider(input2);
        {
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
            assert!(p.no_transition());
        };
        // lut1 , used in low partition dot
        let p = consider(lut1);
        {
            assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
            assert!(
                p.alternative_output_representation == HashSet::from([LOW_PRECISION_PARTITION])
            );
            assert!(p.inputs_transition == vec![None]);
        };
        // lut2
        let p = consider(lut2);
        {
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
            assert!(p.no_transition());
        };
        // dot
        let p = consider(dot);
        {
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
            assert!(p.alternative_output_representation.is_empty());
            assert!(
                p.inputs_transition
                    == vec![
                        Some(Transition::Additional {
                            src_partition: HIGH_PRECISION_PARTITION
                        }),
                        None
                    ]
            );
        };
    }

    #[test]
    fn test_rounded_v3_first_layer_and_second_layer() {
        let acc_precision = 8;
        let precision = 6;
        let mut dag = unparametrized::Dag::new();
        let input1 = dag.add_input(acc_precision, Shape::number());
        let rounded1 = dag.add_expanded_round(input1, precision);
        let lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let rounded2 = dag.add_expanded_round(lut1, precision);
        let lut2 = dag.add_lut(rounded2, FunctionTable::UNKWOWN, acc_precision);
        let partitions = partitionning(&dag);
        let consider = |op_i| &partitions.instrs_partition[op_i];
        // First layer is fully LOW_PRECISION_PARTITION
        for op_i in input1.0..lut1.0 {
            let p = consider(op_i);
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
            assert!(p.no_transition());
        }
        // First lut is HIGH_PRECISION_PARTITION and immedialtely converted to LOW_PRECISION_PARTITION
        let p = consider(lut1.0);
        {
            assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
            assert!(
                p.alternative_output_representation == HashSet::from([LOW_PRECISION_PARTITION])
            );
            assert!(
                p.inputs_transition
                    == vec![Some(Transition::Internal {
                        src_partition: LOW_PRECISION_PARTITION
                    })]
            );
        };
        for op_i in (lut1.0 + 1)..lut2.0 {
            let p = consider(op_i);
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
        }
        let p = consider(lut2.0);
        {
            assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
            assert!(p.alternative_output_representation.is_empty());
            assert!(
                p.inputs_transition
                    == vec![Some(Transition::Internal {
                        src_partition: LOW_PRECISION_PARTITION
                    })]
            );
        };
    }

    #[test]
    fn test_rounded_v3_classic_first_layer_second_layer() {
        let acc_precision = 8;
        let precision = 6;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let first_layer = free_input1.0..=input1.0;
        let rounded1 = dag.add_expanded_round(input1, precision);
        let rounded_layer: Vec<_> = ((input1.0 + 1)..rounded1.0).collect();
        let lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let partitions = partitionning(&dag);
        let consider = |op_i: usize| &partitions.instrs_partition[op_i];

        // First layer is fully HIGH_PRECISION_PARTITION
        for op_i in first_layer {
            let p = consider(op_i);
            assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
        }
        // input is converted with a fast keyswitch to LOW_PRECISION_PARTITION
        let p = consider(input1.0);
        assert!(p.alternative_output_representation == HashSet::from([LOW_PRECISION_PARTITION]));
        let read_converted = Some(Transition::Additional {
            src_partition: HIGH_PRECISION_PARTITION,
        });

        // Second layer, rounded part is LOW_PRECISION_PARTITION
        for &op_i in &rounded_layer {
            let p = consider(op_i);
            assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
        }
        // and use read the conversion result
        let mut first_bit_extract_verified = false;
        let mut first_bit_erase_verified = false;
        for &op_i in &rounded_layer {
            let p = consider(op_i);
            if let Op::Dot { weights, .. } = &dag.operators[op_i] {
                let first_bit_extract = weights.values == [256] && !first_bit_extract_verified;
                let first_bit_erase = weights.values == [1, -1] && !first_bit_erase_verified;
                if first_bit_extract || first_bit_erase {
                    assert!(p.inputs_transition[0] == read_converted);
                }
                first_bit_extract_verified = first_bit_extract_verified || first_bit_extract;
                first_bit_erase_verified = first_bit_erase_verified || first_bit_erase;
            };
        }
        assert!(first_bit_extract_verified);
        assert!(first_bit_erase_verified);
        // Second layer, lut part is HIGH_PRECISION_PARTITION
        // and use an internal conversion
        let p = consider(lut1.0);
        assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
        assert!(
            p.inputs_transition[0]
                == Some(Transition::Internal {
                    src_partition: LOW_PRECISION_PARTITION
                })
        );
    }

    #[test]
    fn test_rounded_v1_classic_first_layer_second_layer() {
        let acc_precision = 8;
        let precision = 6;
        let mut dag = unparametrized::Dag::new();
        let free_input1 = dag.add_input(precision, Shape::number());
        let input1 = dag.add_lut(free_input1, FunctionTable::UNKWOWN, acc_precision);
        let first_layer = free_input1.0..=input1.0;
        let rounded1 = dag.add_expanded_round(input1, precision);
        let rounded_layer = (input1.0 + 1)..rounded1.0;
        let _lut1 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, acc_precision);
        let partitions =
            partitionning_with_preferred(&dag, &default_p_cut(), HIGH_PRECISION_PARTITION);
        show_partitionning(&dag, &partitions.instrs_partition);
        let consider = |op_i: usize| &partitions.instrs_partition[op_i];

        // First layer is fully HIGH_PRECISION_PARTITION
        for op_i in first_layer {
            assert!(consider(op_i).instruction_partition == HIGH_PRECISION_PARTITION);
        }
        // input is converted with a fast keyswitch to LOW_PRECISION_PARTITION
        assert!(consider(input1.0)
            .alternative_output_representation
            .is_empty());
        let read_converted = Some(Transition::Additional {
            src_partition: LOW_PRECISION_PARTITION,
        });

        // Second layer, rounded part is mostly HIGH_PRECISION_PARTITION
        // Only the Lut is post-converted
        for op_i in rounded_layer {
            let p = consider(op_i);
            match &dag.operators[op_i] {
                Op::Lut { .. } => {
                    assert!(p.instruction_partition == LOW_PRECISION_PARTITION);
                    assert!(
                        p.alternative_output_representation
                            == HashSet::from([HIGH_PRECISION_PARTITION])
                    );
                }
                Op::Dot { weights, .. } => {
                    assert!(p.instruction_partition == HIGH_PRECISION_PARTITION);
                    assert!(p.inputs_transition[0].is_none());
                    if weights.values.len() == 2 {
                        assert!(p.inputs_transition[1] == read_converted);
                    }
                }
                _ => assert!(p.instruction_partition == HIGH_PRECISION_PARTITION),
            }
        }
    }
}
