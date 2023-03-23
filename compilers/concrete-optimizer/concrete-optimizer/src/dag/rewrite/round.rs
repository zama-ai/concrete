use crate::dag::operator::{Operator, OperatorIndex};
use crate::dag::unparametrized::OperationDag;

use super::regen::regen;

fn regen_round(_: usize, op: &Operator, dag: &mut OperationDag) -> Option<OperatorIndex> {
    match *op {
        Operator::Round {
            input,
            out_precision,
        } => Some(dag.add_expanded_round(input, out_precision)),
        _ => None,
    }
}

pub(crate) fn expand_round(dag: &OperationDag) -> OperationDag {
    regen(dag, &mut regen_round).0
}

pub(crate) fn expand_round_and_index_map(
    dag: &OperationDag,
) -> (OperationDag, Vec<Vec<OperatorIndex>>) {
    regen(dag, &mut regen_round)
}
