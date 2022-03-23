use crate::global_parameters::{ParameterRanges, ParameterToOperation};
use crate::graph::parameter_indexed::OperatorParameterIndexed;

#[allow(dead_code)]
pub struct AtomicPatternDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameter_ranges: ParameterRanges,
    pub(crate) reverse_map: ParameterToOperation,
}
