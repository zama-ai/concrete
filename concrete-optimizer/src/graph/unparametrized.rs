use super::operator::{Operator, OperatorIndex};
use crate::weight::Weight;

#[derive(Clone)]
#[must_use]
pub struct AtomicPatternDag {
    pub(crate) operators: Vec<Operator<(), ()>>,
}

impl AtomicPatternDag {
    pub const fn new() -> Self {
        Self { operators: vec![] }
    }

    fn add_operator(&mut self, operator: Operator<(), ()>) -> OperatorIndex {
        let operator_index = self.operators.len();

        self.operators.push(operator);

        OperatorIndex(operator_index)
    }

    pub fn add_input(&mut self, out_precision: u8) -> OperatorIndex {
        self.add_operator(Operator::Input {
            out_precision,
            extra_data: (),
        })
    }

    pub fn add_atomic_pattern(
        &mut self,
        in_precision: u8,
        out_precision: u8,
        multisum_inputs: Vec<(Weight, OperatorIndex)>,
    ) -> OperatorIndex {
        self.add_operator(Operator::AtomicPattern {
            in_precision,
            out_precision,
            multisum_inputs,
            extra_data: (),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_creation() {
        let mut graph = AtomicPatternDag::new();

        let input1 = graph.add_input(1);

        let input2 = graph.add_input(2);

        let atomic_pattern1 =
            graph.add_atomic_pattern(3, 3, vec![(Weight(1), input1), (Weight(2), input2)]);

        let _atomic_pattern2 = graph.add_atomic_pattern(
            4,
            4,
            vec![(Weight(1), atomic_pattern1), (Weight(2), input2)],
        );

        assert_eq!(
            &graph.operators,
            &[
                Operator::Input {
                    out_precision: 1,
                    extra_data: ()
                },
                Operator::Input {
                    out_precision: 2,
                    extra_data: ()
                },
                Operator::AtomicPattern {
                    in_precision: 3,
                    out_precision: 3,
                    multisum_inputs: vec![(Weight(1), input1), (Weight(2), input2)],
                    extra_data: ()
                },
                Operator::AtomicPattern {
                    in_precision: 4,
                    out_precision: 4,
                    multisum_inputs: vec![(Weight(1), atomic_pattern1), (Weight(2), input2)],
                    extra_data: ()
                },
            ]
        );
    }
}
