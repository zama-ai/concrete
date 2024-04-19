use crate::dag::operator::operator::Operator;
use crate::dag::operator::OperatorIndex;
use crate::dag::unparametrized::{Dag, DagBuilder};

fn reindex_op_inputs(op: &Operator, old_index_to_new: &[usize]) -> Operator {
    let mut op = op.clone();
    match &mut op {
        Operator::Input { .. } => (),
        Operator::Lut { input, .. }
        | Operator::UnsafeCast { input, .. }
        | Operator::Round { input, .. } => input.0 = old_index_to_new[input.0],
        Operator::Dot { inputs, .. } | Operator::LevelledOp { inputs, .. } => {
            for input in inputs {
                input.0 = old_index_to_new[input.0];
            }
        }
    };
    op
}

pub(crate) fn regen(
    dag: &Dag,
    f: &mut dyn FnMut(usize, &Operator, &mut DagBuilder<'_>) -> Option<OperatorIndex>,
) -> (Dag, Vec<Vec<OperatorIndex>>) {
    let mut regen_dag = Dag::new();
    let mut old_index_to_new = vec![];
    for (i, op) in dag.operators.iter().enumerate() {
        let op = reindex_op_inputs(op, &old_index_to_new);
        let size = regen_dag.operators.len();
        if let Some(op_i) = f(i, &op, &mut regen_dag.builder(dag.circuit_tags[i].clone())) {
            old_index_to_new.push(op_i.0);
        } else {
            assert!(size == regen_dag.operators.len());
            old_index_to_new.push(regen_dag.len());
            regen_dag.operators.push(op.clone());
            regen_dag.out_precisions.push(dag.out_precisions[i]);
            regen_dag.out_shapes.push(dag.out_shapes[i].clone());
            regen_dag.output_state.push(dag.output_state[i]);
            regen_dag.circuit_tags.push(dag.circuit_tags[i].clone());
            op.get_inputs_iter()
                .for_each(|n| regen_dag.output_state[n.0].transition_use());
        }
    }
    // remap composition
    regen_dag.composition = dag.composition.clone();
    regen_dag.composition.update_index(&old_index_to_new);
    (regen_dag, instructions_multi_map(&old_index_to_new))
}

fn instructions_multi_map(old_index_to_new: &[usize]) -> Vec<Vec<OperatorIndex>> {
    let mut last_new_instr = None;
    let mut result = vec![];
    result.reserve_exact(old_index_to_new.len());
    for &new_instr in old_index_to_new {
        let start_from = last_new_instr.map_or(new_instr, |v: usize| v + 1);
        if start_from <= new_instr {
            result.push((start_from..=new_instr).map(OperatorIndex).collect());
        } else {
            result.push(vec![]);
        }
        last_new_instr = Some(new_instr);
    }
    result
}
