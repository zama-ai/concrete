use crate::dag::operator::{Operator, OperatorIndex};
use crate::dag::unparametrized::{Dag, DagBuilder, DagOperator};

use super::regen::regen;

fn regen_round(
    op: Operator,
    dag_op: &DagOperator<'_>,
    dag_builder: &mut DagBuilder<'_>,
) -> Option<OperatorIndex> {
    match op {
        Operator::Round {
            input,
            out_precision,
        } => Some(dag_builder.add_expanded_round(input, out_precision, dag_op.location.to_owned())),
        _ => None,
    }
}

pub(crate) fn expand_round(dag: &Dag) -> Dag {
    regen(dag, regen_round).0
}

pub(crate) fn expand_round_and_index_map(dag: &Dag) -> (Dag, Vec<Vec<OperatorIndex>>) {
    regen(dag, regen_round)
}
