use crate::dag::parameter_indexed::OperatorParameterIndexed;
use crate::global_parameters::{ParameterRanges, ParameterToOperation};

#[allow(dead_code)]
pub struct OperationDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameter_ranges: ParameterRanges,
    pub(crate) reverse_map: ParameterToOperation,
}
