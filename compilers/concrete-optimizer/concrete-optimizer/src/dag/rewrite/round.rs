use crate::dag::operator::{Operator, OperatorIndex};
use crate::dag::unparametrized::Dag;

use super::regen::regen;

fn regen_round(_: usize, op: &Operator, dag: &mut Dag) -> Option<OperatorIndex> {
    match *op {
        Operator::Round {
            input,
            out_precision,
        } => Some(dag.add_expanded_round(input, out_precision)),
        _ => None,
    }
}

pub(crate) fn expand_round(dag: &Dag) -> Dag {
    regen(dag, &mut regen_round).0
}

pub(crate) fn expand_round_and_index_map(dag: &Dag) -> (Dag, Vec<Vec<OperatorIndex>>) {
    regen(dag, &mut regen_round)
}
