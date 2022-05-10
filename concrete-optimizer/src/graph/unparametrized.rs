use crate::graph::operator::{
    FunctionTable, LevelledComplexity, Operator, OperatorIndex, Shape, Weights,
};

pub(crate) type UnparameterizedOperator = Operator<(), (), (), ()>;

#[derive(Clone, PartialEq)]
#[must_use]
pub struct OperationDag {
    pub(crate) operators: Vec<UnparameterizedOperator>,
}

impl OperationDag {
    pub const fn new() -> Self {
        Self { operators: vec![] }
    }

    fn add_operator(&mut self, operator: UnparameterizedOperator) -> OperatorIndex {
        let i = self.operators.len();
        self.operators.push(operator);
        OperatorIndex { i }
    }

    pub fn add_input(&mut self, out_precision: u8, out_shape: Shape) -> OperatorIndex {
        self.add_operator(Operator::Input {
            out_precision,
            out_shape,
            extra_data: (),
        })
    }

    pub fn add_lut(&mut self, input: OperatorIndex, table: FunctionTable) -> OperatorIndex {
        self.add_operator(Operator::Lut {
            input,
            table,
            extra_data: (),
        })
    }

    pub fn add_dot(&mut self, inputs: &[OperatorIndex], weights: &Weights) -> OperatorIndex {
        self.add_operator(Operator::Dot {
            inputs: inputs.to_vec(),
            weights: weights.clone(),
            extra_data: (),
        })
    }

    pub fn add_levelled_op(
        &mut self,
        inputs: &[OperatorIndex],
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: Shape,
        comment: &str,
    ) -> OperatorIndex {
        let inputs = inputs.to_vec();
        let comment = comment.to_string();
        let op = Operator::LevelledOp {
            inputs,
            complexity,
            manp,
            out_shape,
            comment,
            extra_data: (),
        };
        self.add_operator(op)
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.operators.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::operator::Shape;

    #[test]
    fn graph_creation() {
        let mut graph = OperationDag::new();

        let input1 = graph.add_input(1, Shape::number());

        let input2 = graph.add_input(2, Shape::number());

        let cpx_add = LevelledComplexity::ADDITION;
        let sum1 = graph.add_levelled_op(&[input1, input2], cpx_add, 1.0, Shape::number(), "sum");

        let lut1 = graph.add_lut(sum1, FunctionTable::UNKWOWN);

        let concat =
            graph.add_levelled_op(&[input1, lut1], cpx_add, 1.0, Shape::vector(2), "concat");

        let dot = graph.add_dot(&[concat], &Weights::vector(&[1, 2]));

        let lut2 = graph.add_lut(dot, FunctionTable::UNKWOWN);

        let ops_index = [input1, input2, sum1, lut1, concat, dot, lut2];
        for (expected_i, op_index) in ops_index.iter().enumerate() {
            assert_eq!(expected_i, op_index.i);
        }

        assert_eq!(
            graph.operators,
            vec![
                Operator::Input {
                    out_precision: 1,
                    out_shape: Shape::number(),
                    extra_data: ()
                },
                Operator::Input {
                    out_precision: 2,
                    out_shape: Shape::number(),
                    extra_data: ()
                },
                Operator::LevelledOp {
                    inputs: vec![input1, input2],
                    complexity: cpx_add,
                    manp: 1.0,
                    out_shape: Shape::number(),
                    comment: "sum".to_string(),
                    extra_data: ()
                },
                Operator::Lut {
                    input: sum1,
                    table: FunctionTable::UNKWOWN,
                    extra_data: ()
                },
                Operator::LevelledOp {
                    inputs: vec![input1, lut1],
                    complexity: cpx_add,
                    manp: 1.0,
                    out_shape: Shape::vector(2),
                    comment: "concat".to_string(),
                    extra_data: ()
                },
                Operator::Dot {
                    inputs: vec![concat],
                    weights: Weights {
                        shape: Shape::vector(2),
                        values: vec![1, 2]
                    },
                    extra_data: ()
                },
                Operator::Lut {
                    input: dot,
                    table: FunctionTable::UNKWOWN,
                    extra_data: ()
                }
            ]
        );
    }
}
