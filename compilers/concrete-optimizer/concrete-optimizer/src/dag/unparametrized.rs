use std::fmt;
use std::fmt::Write;

use crate::dag::operator::{
    dot_kind, DotKind, FunctionTable, LevelledComplexity, Operator, OperatorIndex, Precision,
    Shape, Weights,
};

pub(crate) type UnparameterizedOperator = Operator;

#[derive(Clone, PartialEq, Debug)]
#[must_use]
pub struct OperationDag {
    pub(crate) operators: Vec<UnparameterizedOperator>,
    // Collect all operators ouput shape
    pub(crate) out_shapes: Vec<Shape>,
    // Collect all operators ouput precision
    pub(crate) out_precisions: Vec<Precision>,
    // Collect whether operators are tagged as outputs
    pub(crate) output_tags: Vec<bool>,
}

impl fmt::Display for OperationDag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, op) in self.operators.iter().enumerate() {
            writeln!(f, "%{i} <- {op}")?;
        }
        Ok(())
    }
}

impl OperationDag {
    pub const fn new() -> Self {
        Self {
            operators: vec![],
            out_shapes: vec![],
            out_precisions: vec![],
            output_tags: vec![],
        }
    }

    fn add_operator(&mut self, operator: UnparameterizedOperator) -> OperatorIndex {
        let i = self.operators.len();
        self.out_precisions
            .push(self.infer_out_precision(&operator));
        self.out_shapes.push(self.infer_out_shape(&operator));
        self.operators.push(operator);
        self.output_tags.push(false);
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
        })
    }

    pub fn add_dot(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        weights: impl Into<Weights>,
    ) -> OperatorIndex {
        let inputs = inputs.into();
        let weights = weights.into();
        self.add_operator(Operator::Dot { inputs, weights })
    }

    pub fn add_levelled_op(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: impl Into<Shape>,
        comment: impl Into<String>,
    ) -> OperatorIndex {
        let inputs = inputs.into();
        let out_shape = out_shape.into();
        let comment = comment.into();
        let op = Operator::LevelledOp {
            inputs,
            complexity,
            manp,
            out_shape,
            comment,
        };
        self.add_operator(op)
    }

    pub fn add_unsafe_cast(
        &mut self,
        input: OperatorIndex,
        out_precision: Precision,
    ) -> OperatorIndex {
        let input_precision = self.out_precisions[input.i];
        if input_precision == out_precision {
            return input;
        }
        self.add_operator(Operator::UnsafeCast {
            input,
            out_precision,
        })
    }

    pub fn add_round_op(
        &mut self,
        input: OperatorIndex,
        rounded_precision: Precision,
    ) -> OperatorIndex {
        let in_precision = self.out_precisions[input.i];
        assert!(rounded_precision <= in_precision);
        self.add_operator(Operator::Round {
            input,
            out_precision: rounded_precision,
        })
    }

    pub fn tag_operator_as_output(&mut self, operator: OperatorIndex) {
        assert!(operator.i < self.len());
        self.output_tags[operator.i] = true;
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

    fn add_shift_left_lsb_to_msb_no_padding(&mut self, input: OperatorIndex) -> OperatorIndex {
        // Convert any input to simple 1bit msb replacing the padding
        // For now encoding is not explicit, so 1 bit content without padding <=> 0 bit content with padding.
        let in_precision = self.out_precisions[input.i];
        let shift_factor = Weights::number(1 << (in_precision as i64));
        let lsb_as_msb = self.add_dot([input], shift_factor);
        self.add_unsafe_cast(lsb_as_msb, 0 as Precision)
    }

    fn add_lut_1bit_no_padding(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
    ) -> OperatorIndex {
        // For now encoding is not explicit, so 1 bit content without padding <=> 0 bit content with padding.
        let in_precision = self.out_precisions[input.i];
        assert!(in_precision == 0);
        // An add after with a clear constant is skipped here as it doesn't change noise handling.
        self.add_lut(input, table, out_precision)
    }

    fn add_shift_right_msb_no_padding_to_lsb(
        &mut self,
        input: OperatorIndex,
        out_precision: Precision,
    ) -> OperatorIndex {
        // Convert simple 1 bit msb to a nbit with zero padding
        let to_nbits_padded = FunctionTable::UNKWOWN;
        self.add_lut_1bit_no_padding(input, to_nbits_padded, out_precision)
    }

    fn add_isolate_lowest_bit(&mut self, input: OperatorIndex) -> OperatorIndex {
        // The lowest bit is converted to a cyphertext of same precision as input.
        // Introduce a pbs of input precision but this precision is only used on 1 levelled op and converted to lower precision
        // Noise is reduced by a pbs.
        let out_precision = self.out_precisions[input.i];
        let lsb_as_msb = self.add_shift_left_lsb_to_msb_no_padding(input);
        self.add_shift_right_msb_no_padding_to_lsb(lsb_as_msb, out_precision)
    }

    pub fn add_truncate_1_bit(&mut self, input: OperatorIndex) -> OperatorIndex {
        // Reset a bit.
        // ex: 10110 is truncated to 1011, 10111 is truncated to 1011
        let in_precision = self.out_precisions[input.i];
        let lowest_bit = self.add_isolate_lowest_bit(input);
        let bit_cleared = self.add_dot([input, lowest_bit], [1, -1]);
        self.add_unsafe_cast(bit_cleared, in_precision - 1)
    }

    pub fn add_expanded_round(
        &mut self,
        input: OperatorIndex,
        rounded_precision: Precision,
    ) -> OperatorIndex {
        // Round such that the ouput has precision out_precision.
        // We round by adding 2**(removed_precision - 1) to the last remaining bit to clear (this step is a no-op).
        // Than all lower bits are cleared.
        // Note: this is a simplified graph, some constant additions are missing without consequence on crypto parameter choice.
        // Note: reset and rounds could be done by 4, 3, 2 and 1 bits groups for efficiency.
        //       bit efficiency is better for 4 precision then 3, but the feasability is lower for high noise
        let in_precision = self.out_precisions[input.i];
        assert!(rounded_precision <= in_precision);
        if in_precision == rounded_precision {
            return input;
        }
        // Add rounding constant, this is a represented as non-op since it doesn't influence crypto parameters.
        let mut rounded = input;
        // The rounded is in high precision with garbage lowest bits
        let bits_to_truncate = in_precision - rounded_precision;
        for _ in 1..=bits_to_truncate as i64 {
            rounded = self.add_truncate_1_bit(rounded);
        }
        rounded
    }

    pub fn add_expanded_rounded_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        rounded_precision: Precision,
        out_precision: Precision,
    ) -> OperatorIndex {
        // note: this is a simplified graph, some constant additions are missing without consequence on crypto parameter choice.
        let rounded = self.add_expanded_round(input, rounded_precision);
        self.add_lut(rounded, table, out_precision)
    }

    pub fn add_rounded_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        rounded_precision: Precision,
        out_precision: Precision,
    ) -> OperatorIndex {
        let rounded = self.add_round_op(input, rounded_precision);
        self.add_lut(rounded, table, out_precision)
    }

    /// Concatenates two dags into a single one (with two disconnected clusters).
    pub fn concat(&mut self, other: &Self) {
        let length = self.len();
        self.operators.extend(other.operators.iter().cloned());
        self.out_precisions.extend(other.out_precisions.iter());
        self.out_shapes.extend(other.out_shapes.iter().cloned());
        self.output_tags.extend(other.output_tags.iter());
        self.operators[length..]
            .iter_mut()
            .for_each(|node| match node {
                Operator::Lut { ref mut input, .. }
                | Operator::UnsafeCast { ref mut input, .. }
                | Operator::Round { ref mut input, .. } => {
                    input.i += length;
                }
                Operator::Dot { ref mut inputs, .. }
                | Operator::LevelledOp { ref mut inputs, .. } => {
                    inputs.iter_mut().for_each(|inp| inp.i += length);
                }
                _ => (),
            });
    }

    /// Returns an iterator over input nodes indices.
    pub(crate) fn get_input_index_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.operators
            .iter()
            .enumerate()
            .filter_map(|(index, op)| match op {
                Operator::Input { .. } => Some(index),
                _ => None,
            })
    }

    /// If no outputs were declared, automatically tag final nodes as outputs.
    #[allow(unused)]
    pub(crate) fn detect_outputs(&mut self) {
        assert!(!self.is_output_tagged());
        self.output_tags = vec![true; self.len()];
        self.operators
            .iter()
            .flat_map(|op| op.get_inputs_iter())
            .for_each(|op| self.output_tags[op.i] = false);
    }

    fn is_output_tagged(&self) -> bool {
        self.output_tags
            .iter()
            .copied()
            .reduce(|a, b| a || b)
            .unwrap()
    }

    /// Returns an iterator over output nodes indices.
    pub(crate) fn get_output_index_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.output_tags
            .iter()
            .enumerate()
            .filter_map(|(index, is_output)| is_output.then_some(index))
    }

    /// Returns whether the node is tagged as output.
    pub(crate) fn is_output_node(&self, oid: usize) -> bool {
        self.output_tags[oid]
    }

    fn infer_out_shape(&self, op: &UnparameterizedOperator) -> Shape {
        match op {
            Operator::Input { out_shape, .. } | Operator::LevelledOp { out_shape, .. } => {
                out_shape.clone()
            }
            Operator::Lut { input, .. }
            | Operator::UnsafeCast { input, .. }
            | Operator::Round { input, .. } => self.out_shapes[input.i].clone(),
            Operator::Dot {
                inputs, weights, ..
            } => {
                let input_shape = self.out_shapes[inputs[0].i].clone();
                let kind = dot_kind(inputs.len() as u64, &input_shape, weights);
                match kind {
                    DotKind::Simple | DotKind::Tensor | DotKind::CompatibleTensor => {
                        Shape::number()
                    }
                    DotKind::Broadcast { shape } => shape,
                    DotKind::Unsupported { .. } => {
                        let weights_shape = &weights.shape;

                        println!();
                        println!();
                        println!("Error diagnostic on dot operation:");
                        println!(
                            "Incompatible operands: <{input_shape:?}> DOT <{weights_shape:?}>"
                        );
                        println!();
                        panic!("Unsupported or invalid dot operation")
                    }
                }
            }
        }
    }

    fn infer_out_precision(&self, op: &UnparameterizedOperator) -> Precision {
        match op {
            Operator::Input { out_precision, .. }
            | Operator::Lut { out_precision, .. }
            | Operator::UnsafeCast { out_precision, .. }
            | Operator::Round { out_precision, .. } => *out_precision,
            Operator::Dot { inputs, .. } | Operator::LevelledOp { inputs, .. } => {
                self.out_precisions[inputs[0].i]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::operator::Shape;

    #[test]
    fn graph_concat() {
        let mut graph1 = OperationDag::new();
        let a = graph1.add_input(1, Shape::number());
        let b = graph1.add_input(1, Shape::number());
        let c = graph1.add_dot([a, b], [1, 1]);
        let _d = graph1.add_lut(c, FunctionTable::UNKWOWN, 1);
        let mut graph2 = OperationDag::new();
        let a = graph2.add_input(2, Shape::number());
        let b = graph2.add_input(2, Shape::number());
        let c = graph2.add_dot([a, b], [2, 2]);
        let _d = graph2.add_lut(c, FunctionTable::UNKWOWN, 2);
        graph1.concat(&graph2);

        let mut graph3 = OperationDag::new();
        let a = graph3.add_input(1, Shape::number());
        let b = graph3.add_input(1, Shape::number());
        let c = graph3.add_dot([a, b], [1, 1]);
        let _d = graph3.add_lut(c, FunctionTable::UNKWOWN, 1);
        let a = graph3.add_input(2, Shape::number());
        let b = graph3.add_input(2, Shape::number());
        let c = graph3.add_dot([a, b], [2, 2]);
        let _d = graph3.add_lut(c, FunctionTable::UNKWOWN, 2);

        assert_eq!(graph1, graph3);
    }

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
                },
                Operator::Input {
                    out_precision: 2,
                    out_shape: Shape::number(),
                },
                Operator::LevelledOp {
                    inputs: vec![input1, input2],
                    complexity: cpx_add,
                    manp: 1.0,
                    out_shape: Shape::number(),
                    comment: "sum".to_string(),
                },
                Operator::Lut {
                    input: sum1,
                    table: FunctionTable::UNKWOWN,
                    out_precision: 1,
                },
                Operator::LevelledOp {
                    inputs: vec![input1, lut1],
                    complexity: cpx_add,
                    manp: 1.0,
                    out_shape: Shape::vector(2),
                    comment: "concat".to_string(),
                },
                Operator::Dot {
                    inputs: vec![concat],
                    weights: Weights {
                        shape: Shape::vector(2),
                        values: vec![1, 2]
                    },
                },
                Operator::Lut {
                    input: dot,
                    table: FunctionTable::UNKWOWN,
                    out_precision: 2,
                }
            ]
        );
    }

    #[test]
    fn test_rounded_lut() {
        let mut graph = OperationDag::new();
        let out_precision = 5;
        let rounded_precision = 2;
        let input1 = graph.add_input(out_precision, Shape::number());
        _ = graph.add_expanded_rounded_lut(
            input1,
            FunctionTable::UNKWOWN,
            rounded_precision,
            out_precision,
        );
        let expecteds = [
            Operator::Input {
                out_precision,
                out_shape: Shape::number(),
            },
            // The rounding addition skipped, it's a no-op wrt crypto parameter
            // Clear: cleared = input - bit0
            //// Extract bit
            Operator::Dot {
                inputs: vec![input1],
                weights: Weights::number(1 << 5),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 1 },
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex { i: 2 },
                table: FunctionTable::UNKWOWN,
                out_precision: 5,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![input1, OperatorIndex { i: 3 }],
                weights: Weights::vector([1, -1]),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 4 },
                out_precision: 4,
            },
            // Clear: cleared = input - bit0 - bit1
            //// Extract bit
            Operator::Dot {
                inputs: vec![OperatorIndex { i: 5 }],
                weights: Weights::number(1 << 4),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 6 },
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex { i: 7 },
                table: FunctionTable::UNKWOWN,
                out_precision: 4,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![OperatorIndex { i: 5 }, OperatorIndex { i: 8 }],
                weights: Weights::vector([1, -1]),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 9 },
                out_precision: 3,
            },
            // Clear: cleared = input - bit0 - bit1 - bit2
            //// Extract bit
            Operator::Dot {
                inputs: vec![OperatorIndex { i: 10 }],
                weights: Weights::number(1 << 3),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 11 },
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex { i: 12 },
                table: FunctionTable::UNKWOWN,
                out_precision: 3,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![OperatorIndex { i: 10 }, OperatorIndex { i: 13 }],
                weights: Weights::vector([1, -1]),
            },
            Operator::UnsafeCast {
                input: OperatorIndex { i: 14 },
                out_precision: 2,
            },
            // Lut on rounded precision
            Operator::Lut {
                input: OperatorIndex { i: 15 },
                table: FunctionTable::UNKWOWN,
                out_precision: 5,
            },
        ];
        assert_eq!(expecteds.len(), graph.operators.len());
        for (i, (expected, actual)) in std::iter::zip(expecteds, graph.operators).enumerate() {
            assert_eq!(expected, actual, "{i}-th operation");
        }
    }
}
