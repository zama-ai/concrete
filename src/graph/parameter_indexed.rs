use crate::global_parameters::{ParameterCount, ParameterToOperation};
use crate::parameters::{AtomicPatternParameters, InputParameter};

use super::operator::Operator;

type Index = usize;

pub struct AtomicPatternDag {
    pub(crate) operators: Vec<
        Operator<InputParameter<usize>, AtomicPatternParameters<Index, Index, Index, Index, Index>>,
    >,
    pub(crate) parameters_count: ParameterCount,
    pub(crate) reverse_map: ParameterToOperation,
}
