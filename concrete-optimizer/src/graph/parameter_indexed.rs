use crate::global_parameters::{ParameterCount, ParameterToOperation};

use super::operator::Operator;

pub struct InputParameterIndexed {
    pub lwe_dimension_index: usize,
}

#[derive(Copy, Clone)]
pub struct AtomicPatternParametersIndexed {
    pub input_lwe_dimensionlwe_dimension_index: usize,
    pub ks_decomposition_parameter_index: usize,
    pub internal_lwe_dimension_index: usize,
    pub br_decomposition_parameter_index: usize,
    pub output_glwe_params_index: usize,
}

pub(crate) type OperatorParameterIndexed =
    Operator<InputParameterIndexed, AtomicPatternParametersIndexed>;

pub struct AtomicPatternDag {
    pub(crate) operators: Vec<OperatorParameterIndexed>,
    pub(crate) parameters_count: ParameterCount,
    pub(crate) reverse_map: ParameterToOperation,
}
