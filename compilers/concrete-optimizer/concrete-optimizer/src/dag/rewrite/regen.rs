use crate::dag::operator::operator::Operator;
use crate::dag::operator::OperatorIndex;
use crate::dag::unparametrized::{Dag, DagBuilder, DagOperator};

fn reindex_op_inputs(op: &Operator, old_index_to_new: &[usize]) -> Operator {
    let mut op = op.clone();
    match &mut op {
        Operator::Input { .. } | Operator::ZeroNoise { .. } => (),
        Operator::Lut { input, .. }
        | Operator::UnsafeCast { input, .. }
        | Operator::Round { input, .. }
        | Operator::ChangePartition { input, .. } => input.0 = old_index_to_new[input.0],
        Operator::Dot { inputs, .. }
        | Operator::LinearNoise { inputs, .. }
        | Operator::MaxNoise { inputs, .. } => {
            for input in inputs {
                input.0 = old_index_to_new[input.0];
            }
        }
    };
    op
}

pub(crate) fn regen<F>(dag: &Dag, f: F) -> (Dag, Vec<Vec<OperatorIndex>>)
where
    F: Fn(Operator, &DagOperator<'_>, &mut DagBuilder<'_>) -> Option<OperatorIndex>,
{
    let mut regen_dag = Dag::new();
    let mut old_index_to_new = vec![];
    for dag_op in dag.get_operators_iter() {
        let new_op = reindex_op_inputs(dag_op.operator, &old_index_to_new);
        let size = regen_dag.operators.len();
        let mut builder = regen_dag.builder(dag_op.circuit_tag);
        if let Some(op_i) = f(new_op.clone(), &dag_op, &mut builder) {
            old_index_to_new.push(op_i.0);
        } else {
            assert!(size == regen_dag.operators.len());
            old_index_to_new.push(regen_dag.len());
            regen_dag.operators.push(new_op.clone());
            regen_dag.out_precisions.push(*dag_op.precision);
            regen_dag.out_shapes.push(dag_op.shape.to_owned());
            regen_dag.output_state.push(*dag_op.output_state);
            regen_dag.circuit_tags.push(dag_op.circuit_tag.to_owned());
            regen_dag.locations.push(dag_op.location.to_owned());
            new_op
                .get_inputs_iter()
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
