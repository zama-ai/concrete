use crate::weight::Weight;

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Operator<InputExtraData, AtomicPatternExtraData> {
    Input {
        out_precision: u8,
        extra_data: InputExtraData,
    },
    AtomicPattern {
        in_precision: u8,
        out_precision: u8,
        multisum_inputs: Vec<(Weight, OperatorIndex)>,
        extra_data: AtomicPatternExtraData,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OperatorIndex(pub usize);
