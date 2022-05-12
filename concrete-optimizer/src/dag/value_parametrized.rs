use crate::dag::parameter_indexed::OperatorParameterIndexed;
use crate::global_parameters::{ParameterToOperation, ParameterValues};

#[allow(dead_code)]
pub struct OperationDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameter_ranges: ParameterValues,
    pub(crate) reverse_map: ParameterToOperation,
}
