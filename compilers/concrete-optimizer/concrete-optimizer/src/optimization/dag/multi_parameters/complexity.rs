use std::{fmt, ops::Add};

use super::{
    partitions::PartitionIndex,
    symbolic::{fast_keyswitch, keyswitch, Symbol, SymbolArray, SymbolMap, SymbolScheme},
};

/// A structure storing the number of times an fhe operation gets executed in a circuit.
#[derive(Clone, Debug)]
pub struct OperationsCount(pub(super) SymbolMap<usize>);

impl Add<OperationsCount> for OperationsCount {
    type Output = OperationsCount;

    fn add(self, rhs: OperationsCount) -> Self::Output {
        let mut output = self;
        for (s, v) in rhs.0.into_iter() {
            output.0.update(s, |a| a + v);
        }
        output
    }
}

impl fmt::Display for OperationsCount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt_with(f, "+", "¢")
    }
}

/// An ensemble of costs associated with fhe operation symbols.
#[derive(Clone, Debug)]
pub struct ComplexityValues(SymbolArray<f64>);

impl ComplexityValues {
    /// Returns an empty set of cost values.
    pub fn from_scheme(scheme: &SymbolScheme) -> ComplexityValues {
        ComplexityValues(SymbolArray::from_scheme(scheme))
    }

    /// Sets the cost associated with an fhe operation symbol.
    pub fn set_cost(&mut self, source: Symbol, value: f64) {
        self.0.set(&source, value);
    }
}

/// A complexity expression is a sum of complexity terms associating operation
/// symbols with the number of time they gets executed in the circuit.
#[derive(Clone, Debug)]
pub struct ComplexityEvaluator(SymbolArray<usize>);

impl ComplexityEvaluator {
    /// Creates a complexity expression from a set of operation counts.
    pub fn from_scheme_and_counts(
        scheme: &SymbolScheme,
        counts: &OperationsCount,
    ) -> ComplexityEvaluator {
        Self(SymbolArray::from_scheme_and_map(scheme, &counts.0))
    }

    pub fn scheme(&self) -> &SymbolScheme {
        self.0.scheme()
    }

    /// Evaluates the total cost expression on a set of cost values.
    pub fn evaluate_total_cost(&self, costs: &ComplexityValues) -> f64 {
        self.0
            .iter()
            .zip(costs.0.iter())
            .fold(0.0, |acc, (n_ops, cost)| acc + (*n_ops as f64) * *cost)
    }

    /// Evaluates the max ks cost expression on a set of cost values.
    pub fn evaluate_ks_max_cost(
        &self,
        complexity_cut: f64,
        costs: &ComplexityValues,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> f64 {
        let actual_ks_cost = costs.0.get(&keyswitch(src_partition, dst_partition));
        let ks_coeff = self.0.get(&keyswitch(src_partition, dst_partition));
        let actual_complexity =
            self.evaluate_total_cost(costs) - (*ks_coeff as f64) * actual_ks_cost;
        (complexity_cut - actual_complexity) / (*ks_coeff as f64)
    }

    /// Evaluates the max fks cost expression on a set of cost values.
    pub fn evaluate_fks_max_cost(
        &self,
        complexity_cut: f64,
        costs: &ComplexityValues,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> f64 {
        let actual_fks_cost = costs.0.get(&fast_keyswitch(src_partition, dst_partition));
        let fks_coeff = self.0.get(&fast_keyswitch(src_partition, dst_partition));
        let actual_complexity =
            self.evaluate_total_cost(costs) - (*fks_coeff as f64) * actual_fks_cost;
        (complexity_cut - actual_complexity) / (*fks_coeff as f64)
    }
}
