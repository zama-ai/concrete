use crate::dag::operator::{
    FunctionTable, LevelledComplexity, Operator, OperatorIndex, Precision, Shape, Weights,
};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use super::operator::dot_kind::{dot_kind, DotKind};

/// The name of the default. Used when adding operations directly on the dag instead of via a
/// builder.
const DEFAULT_CIRCUIT: &str = "_";

/// A state machine to define if an operator is used as output to a circuit.
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum OutputState {
    /// The operator was created and neither used as input to another operator, nor tagged as output
    /// explicitly. It is considered an output.
    Unused,
    // The operator was created, never explicitly tagged and eventually, used as input to another
    // operator. It is not considered an output.
    Used,
    // The operator was explicitly tagged as output. It is considered an output.
    Tagged,
}

impl OutputState {
    /// Creates a fresh output state.
    pub(crate) fn new() -> Self {
        Self::Unused
    }

    /// Transitions from an `unused` state to a `used` state.
    pub(crate) fn transition_use(&mut self) {
        if matches!(self, Self::Unused) {
            *self = Self::Used;
        }
    }

    /// Transitions from any state to a `tagged` state.
    pub(crate) fn transition_tag(&mut self) {
        *self = Self::Tagged;
    }

    /// Tells whether a state corresponds to the operator being an output
    pub(crate) fn is_output(self) -> bool {
        match self {
            Self::Tagged | Self::Unused => true,
            Self::Used => false,
        }
    }
}

/// A type referencing every informations related to an operator of the dag.
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct DagOperator<'dag> {
    pub id: OperatorIndex,
    pub dag: &'dag Dag,
    pub operator: &'dag Operator,
    pub shape: &'dag Shape,
    pub precision: &'dag Precision,
    pub output_state: &'dag OutputState,
    pub circuit_tag: &'dag String,
}

impl<'dag> DagOperator<'dag> {
    /// Returns if the operator is an input.
    pub fn is_input(&self) -> bool {
        matches!(self.operator, Operator::Input { .. })
    }

    /// Returns if the operator is an output.
    pub fn is_output(&self) -> bool {
        self.output_state.is_output()
    }

    /// Returns an iterator over the operators used as input to this operator.
    pub fn get_inputs_iter(&self) -> impl Iterator<Item = DagOperator<'dag>> {
        self.operator
            .get_inputs_iter()
            .map(|id| self.dag.get_operator(*id))
    }
}

/// A structure referencing the operators associated to a particular circuit.
#[derive(Debug, Clone)]
pub struct DagCircuit<'dag> {
    pub(crate) dag: &'dag Dag,
    pub(crate) ids: Vec<OperatorIndex>,
    pub(crate) circuit: String,
}

impl<'dag> DagCircuit<'dag> {
    /// Returns an iterator over the operators of this circuit.
    pub fn get_operators_iter(&self) -> impl Iterator<Item = DagOperator<'dag>> + '_ {
        self.ids.iter().map(|id| self.dag.get_operator(*id))
    }

    /// Returns an iterator over the circuit's input operators.
    #[allow(unused)]
    pub fn get_input_operators_iter(&self) -> impl Iterator<Item = DagOperator<'dag>> + '_ {
        self.get_operators_iter().filter(DagOperator::is_input)
    }

    /// Returns an iterator over the circuit's output operators.
    #[allow(unused)]
    pub fn get_output_operators_iter(&self) -> impl Iterator<Item = DagOperator<'dag>> + '_ {
        self.get_operators_iter().filter(DagOperator::is_output)
    }
}

impl<'dag> fmt::Display for DagCircuit<'dag> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for op in self.get_operators_iter() {
            writeln!(f, "{} <- {:?}", op.id, op.operator)?;
        }
        Ok(())
    }
}

/// A type allowing build a circuit in a dag.
///
/// See [Dag] for more informations on dag building.
#[derive(Debug)]
pub struct DagBuilder<'dag> {
    dag: &'dag mut Dag,
    pub(crate) circuit: String,
}

impl<'dag> DagBuilder<'dag> {
    fn add_operator(&mut self, operator: Operator) -> OperatorIndex {
        debug_assert!(operator
            .get_inputs_iter()
            .all(|id| self.dag.circuit_tags[id.0] == self.circuit));
        let i = self.dag.operators.len();
        self.dag
            .out_precisions
            .push(self.infer_out_precision(&operator));
        self.dag.out_shapes.push(self.infer_out_shape(&operator));
        operator
            .get_inputs_iter()
            .for_each(|id| self.dag.output_state[id.0].transition_use());
        self.dag.operators.push(operator);
        self.dag.output_state.push(OutputState::new());
        self.dag.circuit_tags.push(self.circuit.clone());
        OperatorIndex(i)
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
        // We detect the kind of dot to simplify matching later on.
        let nb_inputs = inputs.len() as u64;
        let input_shape = self.dag.get_operator(inputs[0]).shape;
        self.add_operator(Operator::Dot {
            inputs,
            kind: dot_kind(nb_inputs, input_shape, &weights),
            weights,
        })
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
        let input_precision = self.dag.out_precisions[input.0];
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
        let in_precision = self.dag.out_precisions[input.0];
        assert!(rounded_precision <= in_precision);
        self.add_operator(Operator::Round {
            input,
            out_precision: rounded_precision,
        })
    }

    fn add_shift_left_lsb_to_msb_no_padding(&mut self, input: OperatorIndex) -> OperatorIndex {
        // Convert any input to simple 1bit msb replacing the padding
        // For now encoding is not explicit, so 1 bit content without padding <=> 0 bit content with padding.
        let in_precision = self.dag.out_precisions[input.0];
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
        let in_precision = self.dag.out_precisions[input.0];
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
        // The lowest bit is converted to a ciphertext of same precision as input.
        // Introduce a pbs of input precision but this precision is only used on 1 levelled op and converted to lower precision
        // Noise is reduced by a pbs.
        let out_precision = self.dag.out_precisions[input.0];
        let lsb_as_msb = self.add_shift_left_lsb_to_msb_no_padding(input);
        self.add_shift_right_msb_no_padding_to_lsb(lsb_as_msb, out_precision)
    }

    pub fn add_truncate_1_bit(&mut self, input: OperatorIndex) -> OperatorIndex {
        // Reset a bit.
        // ex: 10110 is truncated to 1011, 10111 is truncated to 1011
        let in_precision = self.dag.out_precisions[input.0];
        let lowest_bit = self.add_isolate_lowest_bit(input);
        let bit_cleared = self.add_dot([input, lowest_bit], [1, -1]);
        self.add_unsafe_cast(bit_cleared, in_precision - 1)
    }

    pub fn add_expanded_round(
        &mut self,
        input: OperatorIndex,
        rounded_precision: Precision,
    ) -> OperatorIndex {
        // Round such that the output has precision out_precision.
        // We round by adding 2**(removed_precision - 1) to the last remaining bit to clear (this step is a no-op).
        // Than all lower bits are cleared.
        // Note: this is a simplified graph, some constant additions are missing without consequence on crypto parameter choice.
        // Note: reset and rounds could be done by 4, 3, 2 and 1 bits groups for efficiency.
        //       bit efficiency is better for 4 precision then 3, but the feasibility is lower for high noise
        let in_precision = self.dag.out_precisions[input.0];
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

    /// Marks an operator as being an output of the circuit.
    ///
    /// # Note:
    ///
    /// Operators without consumers are automatically tagged as output. For operators that are used
    /// as input to another operator, but at the same time are returned by the circuit, they must be
    /// tagged using this method.
    pub fn tag_operator_as_output(&mut self, operator: OperatorIndex) {
        assert!(operator.0 < self.dag.len());
        debug_assert!(self.dag.circuit_tags[operator.0] == self.circuit);
        self.dag.output_state[operator.0].transition_tag();
    }

    pub fn get_circuit(&self) -> DagCircuit<'_> {
        self.dag.get_circuit(&self.circuit)
    }

    fn infer_out_shape(&self, op: &Operator) -> Shape {
        match op {
            Operator::Input { out_shape, .. } | Operator::LevelledOp { out_shape, .. } => {
                out_shape.clone()
            }
            Operator::Lut { input, .. }
            | Operator::UnsafeCast { input, .. }
            | Operator::Round { input, .. } => self.dag.out_shapes[input.0].clone(),
            Operator::Dot {
                kind: DotKind::Simple | DotKind::Tensor | DotKind::CompatibleTensor,
                ..
            } => Shape::number(),
            Operator::Dot {
                kind: DotKind::Broadcast { shape },
                ..
            } => shape.clone(),
            Operator::Dot {
                kind: DotKind::Unsupported { .. },
                weights,
                inputs,
            } => {
                let weights_shape = &weights.shape;
                let input_shape = self.dag.get_operator(inputs[0]).shape;
                println!();
                println!();
                println!("Error diagnostic on dot operation:");
                println!("Incompatible operands: <{input_shape:?}> DOT <{weights_shape:?}>");
                println!();
                panic!("Unsupported or invalid dot operation")
            }
        }
    }

    fn infer_out_precision(&self, op: &Operator) -> Precision {
        match op {
            Operator::Input { out_precision, .. }
            | Operator::Lut { out_precision, .. }
            | Operator::UnsafeCast { out_precision, .. }
            | Operator::Round { out_precision, .. } => *out_precision,
            Operator::Dot { inputs, .. } | Operator::LevelledOp { inputs, .. } => {
                self.dag.out_precisions[inputs[0].0]
            }
        }
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct CompositionRules(HashMap<OperatorIndex, Vec<OperatorIndex>>);

impl CompositionRules {
    pub(crate) fn add(&mut self, from: OperatorIndex, to: OperatorIndex) {
        let _ = self
            .0
            .entry(to)
            .and_modify(|e| e.push(from))
            .or_insert_with(|| [from].into());
    }

    pub(crate) fn update_index(&mut self, old_to_new_map: &[usize]) {
        let mut old_map = HashMap::with_capacity(self.0.capacity());
        std::mem::swap(&mut old_map, &mut self.0);
        for (old_id, mut compositions) in old_map {
            let adapter = |old_id: OperatorIndex| -> OperatorIndex {
                OperatorIndex(old_to_new_map[old_id.0])
            };
            compositions
                .iter_mut()
                .for_each(|from| *from = adapter(*from));
            let _ = self.0.insert(adapter(old_id), compositions);
        }
    }
}

impl IntoIterator for CompositionRules {
    type Item = (OperatorIndex, Vec<OperatorIndex>);
    type IntoIter = std::collections::hash_map::IntoIter<OperatorIndex, Vec<OperatorIndex>>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A type containing a Directed Acyclic Graph of operators.
///
/// This is the major datatype used to encode a module in the optimizer. It is equivalent to an fhe
/// module in the frontend, or to a mlir module in the compiler: it can contain multiple separated
/// circuits which are optimized together.
///
/// To add a new circuit to the dag, one should create a [`DagBuilder`] associated to the circuit
/// using the [`Dag::builder`] method, and use the `DagBuilder::add_*` methods.
///
/// # Note:
///
/// For ease of use in tests, it is also possible to add operators on an anonymous circuit (`_`)
/// directly on a [`Dag`] object itself, using the `Dag::add_*` methods.
#[derive(Clone, PartialEq, Debug)]
#[must_use]
pub struct Dag {
    pub(crate) operators: Vec<Operator>,
    // Collect all operators output shape
    pub(crate) out_shapes: Vec<Shape>,
    // Collect all operators output precision
    pub(crate) out_precisions: Vec<Precision>,
    // Collect the output state of the operators
    pub(crate) output_state: Vec<OutputState>,
    // Collect the circuit the operators are associated with
    pub(crate) circuit_tags: Vec<String>,
    // Composition rules
    pub(crate) composition: CompositionRules,
}

impl fmt::Display for Dag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for op in self.get_operators_iter() {
            writeln!(f, "{} <- {:?}", op.id, op.operator)?;
        }
        Ok(())
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}

impl Dag {
    /// Instantiate a new dag.
    pub fn new() -> Self {
        Self {
            operators: vec![],
            out_shapes: vec![],
            out_precisions: vec![],
            output_state: vec![],
            circuit_tags: vec![],
            composition: CompositionRules::default(),
        }
    }

    /// Returns a builder for the circuit named `circuit`.
    pub fn builder<A: AsRef<str>>(&mut self, circuit: A) -> DagBuilder {
        DagBuilder {
            dag: self,
            circuit: circuit.as_ref().into(),
        }
    }

    pub fn add_input(
        &mut self,
        out_precision: Precision,
        out_shape: impl Into<Shape>,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_input(out_precision, out_shape)
    }

    pub fn add_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_lut(input, table, out_precision)
    }

    pub fn add_dot(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        weights: impl Into<Weights>,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT).add_dot(inputs, weights)
    }

    pub fn add_levelled_op(
        &mut self,
        inputs: impl Into<Vec<OperatorIndex>>,
        complexity: LevelledComplexity,
        manp: f64,
        out_shape: impl Into<Shape>,
        comment: impl Into<String>,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_levelled_op(inputs, complexity, manp, out_shape, comment)
    }

    pub fn add_unsafe_cast(
        &mut self,
        input: OperatorIndex,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_unsafe_cast(input, out_precision)
    }

    pub fn add_round_op(
        &mut self,
        input: OperatorIndex,
        rounded_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_round_op(input, rounded_precision)
    }

    #[allow(unused)]
    fn add_shift_left_lsb_to_msb_no_padding(&mut self, input: OperatorIndex) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_shift_left_lsb_to_msb_no_padding(input)
    }

    #[allow(unused)]
    fn add_lut_1bit_no_padding(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_lut_1bit_no_padding(input, table, out_precision)
    }

    #[allow(unused)]
    fn add_shift_right_msb_no_padding_to_lsb(
        &mut self,
        input: OperatorIndex,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_shift_right_msb_no_padding_to_lsb(input, out_precision)
    }

    #[allow(unused)]
    fn add_isolate_lowest_bit(&mut self, input: OperatorIndex) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT).add_isolate_lowest_bit(input)
    }

    pub fn add_truncate_1_bit(&mut self, input: OperatorIndex) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT).add_truncate_1_bit(input)
    }

    pub fn add_expanded_round(
        &mut self,
        input: OperatorIndex,
        rounded_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT)
            .add_expanded_round(input, rounded_precision)
    }

    pub fn add_expanded_rounded_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        rounded_precision: Precision,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT).add_expanded_rounded_lut(
            input,
            table,
            rounded_precision,
            out_precision,
        )
    }

    pub fn add_rounded_lut(
        &mut self,
        input: OperatorIndex,
        table: FunctionTable,
        rounded_precision: Precision,
        out_precision: Precision,
    ) -> OperatorIndex {
        self.builder(DEFAULT_CIRCUIT).add_expanded_rounded_lut(
            input,
            table,
            rounded_precision,
            out_precision,
        )
    }

    /// Adds a composition rule to the dag.
    pub fn add_composition(&mut self, from: OperatorIndex, to: OperatorIndex) {
        debug_assert!(self.get_operator(from).is_output());
        debug_assert!(self.get_operator(to).is_input());
        self.composition.add(from, to);
    }

    /// Adds a composition rule between every elements of from and every elements of to.
    pub fn add_compositions<A: AsRef<[OperatorIndex]>, B: AsRef<[OperatorIndex]>>(
        &mut self,
        from: A,
        to: B,
    ) {
        for from_i in from.as_ref() {
            for to_i in to.as_ref() {
                self.add_composition(*from_i, *to_i);
            }
        }
    }

    /// Returns whether the dag contains a composition rule.
    pub fn is_composed(&self) -> bool {
        !self.composition.0.is_empty()
    }

    /// Returns an iterator over the operator indices.
    pub fn get_indices_iter(&self) -> impl Iterator<Item = OperatorIndex> {
        (0..self.len()).map(OperatorIndex)
    }

    /// Returns an iterator over the circuits contained in the dag.
    pub fn get_circuits_iter(&self) -> impl Iterator<Item = DagCircuit<'_>> + '_ {
        let mut circuits: HashSet<String> = HashSet::new();
        self.circuit_tags.iter().for_each(|name| {
            let _ = circuits.insert(name.to_owned());
        });
        circuits
            .into_iter()
            .map(|circuit| self.get_circuit(circuit))
    }

    /// Returns a circuit object from its name.
    ///
    /// # Note:
    ///
    /// Panics if no circuit with the given name exist in the dag.
    pub fn get_circuit<A: AsRef<str>>(&self, circuit: A) -> DagCircuit {
        let circuit = circuit.as_ref().to_string();
        assert!(self.circuit_tags.contains(&circuit));
        let ids = self
            .circuit_tags
            .iter()
            .enumerate()
            .filter_map(|(id, circ)| (*circ == circuit).then_some(OperatorIndex(id)))
            .collect();
        DagCircuit {
            dag: self,
            circuit,
            ids,
        }
    }

    /// Returns an iterator over the input operators of the dag.
    #[allow(unused)]
    pub fn get_input_operators_iter(&self) -> impl Iterator<Item = DagOperator<'_>> {
        self.get_indices_iter()
            .map(|i| self.get_operator(i))
            .filter(DagOperator::is_input)
    }

    /// Returns an iterator over the outputs operators of the dag.
    pub fn get_output_operators_iter(&self) -> impl Iterator<Item = DagOperator<'_>> {
        self.get_indices_iter()
            .map(|i| self.get_operator(i))
            .filter(DagOperator::is_output)
    }

    /// Returns an iterator over the operators of the dag.
    pub fn get_operators_iter(&self) -> impl Iterator<Item = DagOperator<'_>> {
        self.get_indices_iter().map(|i| self.get_operator(i))
    }

    /// Returns an operator from its operator index.
    ///
    /// # Note:
    ///
    /// Panics if the operator index is invalid.
    pub fn get_operator(&self, id: OperatorIndex) -> DagOperator<'_> {
        assert!(id.0 < self.len());
        DagOperator {
            dag: self,
            id,
            operator: self.operators.get(id.0).unwrap(),
            shape: self.out_shapes.get(id.0).unwrap(),
            precision: self.out_precisions.get(id.0).unwrap(),
            output_state: self.output_state.get(id.0).unwrap(),
            circuit_tag: self.circuit_tags.get(id.0).unwrap(),
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Marks an operator as being an output of a circuit.
    ///
    /// # Note:
    ///
    /// Operators without consumers are automatically tagged as output. For operators that are used
    /// as input to another operator, but at the same time are returned by the circuit, they must be
    /// tagged using this method.
    pub fn tag_operator_as_output(&mut self, operator: OperatorIndex) {
        assert!(operator.0 < self.len());
        self.output_state[operator.0].transition_tag();
    }

    /// Returns the number of circuits in the dag.
    pub fn get_circuit_count(&self) -> usize {
        self.get_circuits_iter().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_marking() {
        let mut graph = Dag::new();
        let mut builder = graph.builder("main1");
        let a = builder.add_input(1, Shape::number());
        let b = builder.add_input(1, Shape::number());
        builder.tag_operator_as_output(b);
        let _ = builder.add_dot([a, b], [1, 1]);
        assert!(graph.get_operator(b).is_output());
        assert!(!graph.get_operator(a).is_output());
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn graph_builder() {
        let mut graph = Dag::new();
        let mut builder = graph.builder("main1");
        let a = builder.add_input(1, Shape::number());
        let b = builder.add_input(1, Shape::number());
        let c = builder.add_dot([a, b], [1, 1]);
        let _d = builder.add_lut(c, FunctionTable::UNKWOWN, 1);
        let mut builder = graph.builder("main2");
        let e = builder.add_input(2, Shape::number());
        let f = builder.add_input(2, Shape::number());
        let g = builder.add_dot([e, f], [2, 2]);
        let _h = builder.add_lut(g, FunctionTable::UNKWOWN, 2);
        graph.tag_operator_as_output(c);
    }

    #[test]
    fn graph_creation() {
        let mut graph = Dag::new();
        let mut builder = graph.builder("_");
        let input1 = builder.add_input(1, Shape::number());
        let input2 = builder.add_input(2, Shape::number());

        let cpx_add = LevelledComplexity::ADDITION;
        let sum1 = builder.add_levelled_op([input1, input2], cpx_add, 1.0, Shape::number(), "sum");

        let lut1 = builder.add_lut(sum1, FunctionTable::UNKWOWN, 1);

        let concat =
            builder.add_levelled_op([input1, lut1], cpx_add, 1.0, Shape::vector(2), "concat");

        let dot = builder.add_dot([concat], [1, 2]);

        let lut2 = builder.add_lut(dot, FunctionTable::UNKWOWN, 2);

        let ops_index = [input1, input2, sum1, lut1, concat, dot, lut2];
        for (expected_i, op_index) in ops_index.iter().enumerate() {
            assert_eq!(expected_i, op_index.0);
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
                    kind: DotKind::Tensor
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
        let mut graph = Dag::new();
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
                kind: DotKind::Tensor,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(1),
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex(2),
                table: FunctionTable::UNKWOWN,
                out_precision: 5,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![input1, OperatorIndex(3)],
                weights: Weights::vector([1, -1]),
                kind: DotKind::Simple,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(4),
                out_precision: 4,
            },
            // Clear: cleared = input - bit0 - bit1
            //// Extract bit
            Operator::Dot {
                inputs: vec![OperatorIndex(5)],
                weights: Weights::number(1 << 4),
                kind: DotKind::Tensor,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(6),
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex(7),
                table: FunctionTable::UNKWOWN,
                out_precision: 4,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![OperatorIndex(5), OperatorIndex(8)],
                weights: Weights::vector([1, -1]),
                kind: DotKind::Simple,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(9),
                out_precision: 3,
            },
            // Clear: cleared = input - bit0 - bit1 - bit2
            //// Extract bit
            Operator::Dot {
                inputs: vec![OperatorIndex(10)],
                weights: Weights::number(1 << 3),
                kind: DotKind::Tensor,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(11),
                out_precision: 0,
            },
            //// 1 Bit to out_precision
            Operator::Lut {
                input: OperatorIndex(12),
                table: FunctionTable::UNKWOWN,
                out_precision: 3,
            },
            //// Erase bit
            Operator::Dot {
                inputs: vec![OperatorIndex(10), OperatorIndex(13)],
                weights: Weights::vector([1, -1]),
                kind: DotKind::Simple,
            },
            Operator::UnsafeCast {
                input: OperatorIndex(14),
                out_precision: 2,
            },
            // Lut on rounded precision
            Operator::Lut {
                input: OperatorIndex(15),
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
