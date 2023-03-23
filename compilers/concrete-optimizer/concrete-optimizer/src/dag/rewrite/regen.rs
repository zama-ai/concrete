use crate::dag::operator::operator::Operator;
use crate::dag::operator::OperatorIndex;
use crate::dag::unparametrized::OperationDag;

fn reindex_op_inputs(op: &Operator, old_index_to_new: &[usize]) -> Operator {
    let mut op = op.clone();
    match &mut op {
        Operator::Input { .. } => (),
        Operator::Lut { input, .. }
        | Operator::UnsafeCast { input, .. }
        | Operator::Round { input, .. } => input.i = old_index_to_new[input.i],
        Operator::Dot { inputs, .. } | Operator::LevelledOp { inputs, .. } => {
            for input in inputs {
                input.i = old_index_to_new[input.i];
            }
        }
    };
    op
}

pub(crate) fn regen(
    dag: &OperationDag,
    f: &mut dyn FnMut(usize, &Operator, &mut OperationDag) -> Option<OperatorIndex>,
) -> (OperationDag, Vec<Vec<OperatorIndex>>) {
    let mut regen_dag = OperationDag::new();
    let mut old_index_to_new = vec![];
    for (i, op) in dag.operators.iter().enumerate() {
        let op = reindex_op_inputs(op, &old_index_to_new);
        let size = regen_dag.operators.len();
        if let Some(op_i) = f(i, &op, &mut regen_dag) {
            old_index_to_new.push(op_i.i);
        } else {
            assert!(size == regen_dag.operators.len());
            old_index_to_new.push(regen_dag.len());
            regen_dag.operators.push(op.clone());
            regen_dag.out_precisions.push(dag.out_precisions[i]);
            regen_dag.out_shapes.push(dag.out_shapes[i].clone());
        }
    }
    (regen_dag, instructions_multi_map(&old_index_to_new))
}

fn instructions_multi_map(old_index_to_new: &[usize]) -> Vec<Vec<OperatorIndex>> {
    let mut last_new_instr = None;
    let mut result = vec![];
    result.reserve_exact(old_index_to_new.len());
    for &new_instr in old_index_to_new {
        let start_from = last_new_instr.map_or(new_instr, |v: usize| v + 1);
        if start_from <= new_instr {
            result.push(
                (start_from..=new_instr)
                    .map(|i| OperatorIndex { i })
                    .collect(),
            );
        } else {
            result.push(vec![]);
        }
        last_new_instr = Some(new_instr);
    }
    result
}
