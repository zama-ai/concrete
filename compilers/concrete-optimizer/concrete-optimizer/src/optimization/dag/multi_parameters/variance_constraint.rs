use std::fmt;

use crate::dag::operator::Precision;
use crate::optimization::dag::multi_parameters::partitions::PartitionIndex;
use crate::optimization::dag::multi_parameters::symbolic_variance::SymbolicVariance;

#[derive(Clone, Debug)]
pub struct VarianceConstraint {
    pub precision: Precision,
    pub partition: PartitionIndex,
    pub nb_constraints: u64,
    pub safe_variance_bound: f64,
    pub variance: SymbolicVariance,
}

impl fmt::Display for VarianceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} < (2²)**{} ({}bits partition:{} count:{}, dom={})",
            self.variance,
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
    #[allow(clippy::cast_sign_loss)]
    fn dominance_index(&self) -> u64 {
        let max_coeff = self
            .variance
            .coeffs
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        (max_coeff / self.safe_variance_bound).log2().ceil() as u64
    }

    fn dominate_or_equal(&self, other: &Self) -> bool {
        // With BR > Fresh
        let self_var = &self.variance;
        let other_var = &other.variance;
        let self_renorm = other.safe_variance_bound / self.safe_variance_bound;
        let rel_diff =
            |f: &dyn Fn(&SymbolicVariance) -> f64| self_renorm * f(self_var) - f(other_var);
        for partition in PartitionIndex::range(0, self.variance.nb_partitions()) {
            let diffs = [
                rel_diff(&|var| var.coeff_pbs(partition)),
                rel_diff(&|var| var.coeff_pbs(partition) + var.coeff_input(partition)),
                rel_diff(&|var| var.coeff_modulus_switching(partition)),
            ];
            for diff in diffs {
                if diff < 0.0 {
                    return false;
                }
            }
        }
        for src_partition in PartitionIndex::range(0, self.variance.nb_partitions()) {
            for dst_partition in PartitionIndex::range(0, self.variance.nb_partitions()) {
                let diffs = [
                    rel_diff(&|var| var.coeff_keyswitch_to_small(src_partition, dst_partition)),
                    rel_diff(&|var| {
                        var.coeff_partition_keyswitch_to_big(src_partition, dst_partition)
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
