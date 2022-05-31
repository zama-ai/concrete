use std::fmt::Write;

use crate::dag::operator::{
    FunctionTable, LevelledComplexity, Operator, OperatorIndex, Precision, Shape, Weights,
};

pub(crate) type UnparameterizedOperator = Operator<(), (), (), ()>;

#[derive(Clone, PartialEq, Debug)]
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

    pub fn add_input(
        &mut self,
        out_precision: Precision,
        out_shape: impl Into<Shape>,
    ) -> OperatorIndex {
        let out_shape = out_shape.into();
        self.add_operator(Operator::Input {
            out_precision,
            out_shape,
            extra_data: (),
        })
    }

    pub fn add_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.add_operator(Operator::Lut {
            input,
            table,
            out_precision,
            extra_data: (),
        })
    }

    pub fn add_dot(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        weights: impl Into<Weights>,
    ) -> OperatorIndex {
        let inputs = inputs.into();
        let weights = weights.into();
        self.add_operator(Operator::Dot {
            inputs,
            weights,
            extra_data: (),
        })
    }

    pub fn add_levelled_op(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: impl Into<Shape>,
        comment: &str,
    ) -> OperatorIndex {
        let inputs = inputs.into();
        let out_shape = out_shape.into();
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

    pub fn dump(&self) -> String {
        let mut acc = String::new();
        let err_msg = "Optimizer: Can't dump OperationDag";
        writeln!(acc, "Dag:").expect(err_msg);
        for (i, op) in self.operators.iter().enumerate() {
            writeln!(acc, "%{i} <- {op:?}").expect(err_msg);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::operator::Shape;

    #[test]
    fn graph_creation() {
        let mut graph = OperationDag::new();

        let input1 = graph.add_input(1, Shape::number());

        let input2 = graph.add_input(2, Shape::number());

        let cpx_add = LevelledComplexity::ADDITION;
        let sum1 = graph.add_levelled_op([input1, input2], cpx_add, 1.0, Shape::number(), "sum");

        let lut1 = graph.add_lut(sum1, FunctionTable::UNKWOWN, 1);

        let concat =
            graph.add_levelled_op([input1, lut1], cpx_add, 1.0, Shape::vector(2), "concat");

        let dot = graph.add_dot([concat], [1, 2]);

        let lut2 = graph.add_lut(dot, FunctionTable::UNKWOWN, 2);

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
                    out_precision: 1,
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
                    out_precision: 2,
                    extra_data: ()
                }
            ]
        );
    }
}
