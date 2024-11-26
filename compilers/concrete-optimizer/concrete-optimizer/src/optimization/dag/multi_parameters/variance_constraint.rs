use crate::dag::operator::{Location, Precision};
use crate::optimization::dag::multi_parameters::partitions::PartitionIndex;
use std::fmt;

use super::noise_expression::{
    bootstrap_noise, fast_keyswitch_noise, input_noise, keyswitch_noise, modulus_switching_noise,
    NoiseEvaluator, NoiseExpression,
};
use super::symbolic::SymbolScheme;

#[derive(Clone, Debug, PartialEq)]
pub struct VarianceConstraint {
    pub precision: Precision,
    pub partition: PartitionIndex,
    pub nb_partitions: usize,
    pub nb_constraints: u64,
    pub safe_variance_bound: f64,
    pub noise_expression: NoiseExpression,
    pub noise_evaluator: Option<NoiseEvaluator>,
    pub location: Location,
}

impl fmt::Display for VarianceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} < (2²)**{} ({}bits partition:{} count:{}, dom={})",
            self.noise_expression,
            self.safe_variance_bound.log2().round() / 2.0,
            self.precision,
            self.partition,
            self.nb_constraints,
            self.dominance_index()
        )?;
        Ok(())
    }
}

impl VarianceConstraint {
    pub fn init_evaluator(&mut self, scheme: &SymbolScheme) {
        self.noise_evaluator = Some(NoiseEvaluator::from_scheme_and_expression(
            scheme,
            &self.noise_expression,
        ));
    }

    #[allow(clippy::cast_sign_loss)]
    fn dominance_index(&self) -> u64 {
        let max_coeff = self
            .noise_expression
            .terms_iter()
            .map(|t| t.coefficient)
            .fold(0.0, f64::max);
        (max_coeff / self.safe_variance_bound).log2().ceil() as u64
    }

    fn dominate_or_equal(&self, other: &Self) -> bool {
        // With BR > Fresh
        let self_var = &self.noise_expression;
        let other_var = &other.noise_expression;
        let self_renorm = other.safe_variance_bound / self.safe_variance_bound;
        let rel_diff =
            |f: &dyn Fn(&NoiseExpression) -> f64| self_renorm * f(self_var) - f(other_var);
        for partition in PartitionIndex::range(0, self.nb_partitions) {
            let diffs = [
                rel_diff(&|expr| expr.coeff(bootstrap_noise(partition))),
                rel_diff(&|expr| {
                    expr.coeff(bootstrap_noise(partition)) + expr.coeff(input_noise(partition))
                }),
                rel_diff(&|expr| expr.coeff(modulus_switching_noise(partition))),
            ];
            for diff in diffs {
                if diff < 0.0 {
                    return false;
                }
            }
        }
        for src_partition in PartitionIndex::range(0, self.nb_partitions) {
            for dst_partition in PartitionIndex::range(0, self.nb_partitions) {
                let diffs = [
                    rel_diff(&|expr| expr.coeff(keyswitch_noise(src_partition, dst_partition))),
                    rel_diff(&|expr| {
                        expr.coeff(fast_keyswitch_noise(src_partition, dst_partition))
                    }),
                ];
                for diff in diffs {
                    if diff < 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn remove_dominated(constraints: &[Self]) -> Vec<Self> {
        let mut constraints = constraints.to_vec();
        constraints.sort_by_cached_key(Self::dominance_index);
        constraints.reverse();
        let mut dominated = vec![false; constraints.len()];
        for (i, constraint) in constraints.iter().enumerate() {
            if dominated[i] {
                continue;
            }
            for (j, other_constraint) in constraints.iter().enumerate() {
                if j <= i {
                    continue;
                }
                if constraint.dominate_or_equal(other_constraint) {
                    dominated[j] = true;
                } else if other_constraint.dominate_or_equal(constraint) {
                    dominated[i] = true;
                    break;
                }
            }
        }
        let mut result = vec![];
        for (i, c) in constraints.iter().enumerate() {
            if !dominated[i] {
                result.push(c.clone());
            }
        }
        result
    }
}
