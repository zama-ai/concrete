use crate::global_parameters::{ParameterCount, ParameterToOperation};
use crate::parameters::{AtomicPatternParameters, InputParameter};

use super::operator::Operator;

type Index = usize;

type InputParameterIndexed = InputParameter<Index>;

type AtomicPatternParametersIndexed = AtomicPatternParameters<Index, Index, Index, Index, Index>;

pub(crate) type OperatorParameterIndexed =
    Operator<InputParameterIndexed, AtomicPatternParametersIndexed>;

pub struct AtomicPatternDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameters_count: ParameterCount,
    pub(crate) reverse_map: ParameterToOperation,
}
