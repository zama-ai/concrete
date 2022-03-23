use crate::global_parameters::{ParameterToOperation, ParameterValues};
use crate::graph::parameter_indexed::OperatorParameterIndexed;

#[allow(dead_code)]
pub struct AtomicPatternDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameter_ranges: ParameterValues,
    pub(crate) reverse_map: ParameterToOperation,
}
