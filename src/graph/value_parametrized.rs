use crate::global_parameters::{ParameterToOperation, ParameterValues};
use crate::parameters::{AtomicPatternParameters, InputParameter};

use super::operator::Operator;

type Index = usize;

pub struct AtomicPatternDag {
    pub(crate) operators: Vec<
        Operator<InputParameter<usize>, AtomicPatternParameters<Index, Index, Index, Index, Index>>,
    >,
    pub(crate) parameter_ranges: ParameterValues,
    pub(crate) reverse_map: ParameterToOperation,
}
