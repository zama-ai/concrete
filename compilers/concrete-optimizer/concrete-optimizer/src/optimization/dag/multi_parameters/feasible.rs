use crate::noise_estimator::p_error::{combine_errors, repeat_p_error};
use crate::optimization::dag::multi_parameters::variance_constraint::VarianceConstraint;
use crate::optimization::dag::solo_key::analyze::p_error_from_relative_variance;
use crate::utils::f64::f64_dot;

use super::operations_value::OperationsValue;
use super::partitions::PartitionIndex;

#[derive(Clone)]
pub struct Feasible {
    // TODO: move kappa here
    pub constraints: Vec<VarianceConstraint>,
    pub undominated_constraints: Vec<VarianceConstraint>,
    pub kappa: f64, // to convert variance to local probabilities
    pub global_p_error: Option<f64>,
}

impl Feasible {
    pub fn of(constraints: &[VarianceConstraint], kappa: f64, global_p_error: Option<f64>) -> Self {
        let undominated_constraints = VarianceConstraint::remove_dominated(constraints);
        Self {
            kappa,
            constraints: constraints.into(),
            undominated_constraints,
            global_p_error,
        }
    }

    pub fn pbs_max_feasible_variance(
        &self,
        operations_variance: &OperationsValue,
        partition: usize,
    ) -> f64 {
        let pbs_index = operations_variance.index.pbs(partition);
        let actual_pbs_variance = operations_variance.values[pbs_index];

        let mut smallest_pbs_max_variance = std::f64::MAX;

        for constraint in &self.undominated_constraints {
            let pbs_coeff = constraint.variance.coeff_pbs(partition);
            if pbs_coeff == 0.0 {
                continue;
            }
            let actual_variance = f64_dot(operations_variance, &constraint.variance.coeffs)
                - pbs_coeff * actual_pbs_variance;
            let pbs_max_variance = (constraint.safe_variance_bound - actual_variance) / pbs_coeff;
            smallest_pbs_max_variance = smallest_pbs_max_variance.min(pbs_max_variance);
        }
        smallest_pbs_max_variance
    }

    pub fn ks_max_feasible_variance(
        &self,
        operations_variance: &OperationsValue,
        src_partition: usize,
        dst_partition: usize,
    ) -> f64 {
        let ks_index = operations_variance
            .index
            .keyswitch_to_small(src_partition, dst_partition);
        let actual_ks_variance = operations_variance.values[ks_index];

        let mut smallest_ks_max_variance = std::f64::MAX;

        for constraint in &self.undominated_constraints {
            let ks_coeff = constraint
                .variance
                .coeff_keyswitch_to_small(src_partition, dst_partition);
            if ks_coeff == 0.0 {
                continue;
            }
            let actual_variance = f64_dot(operations_variance, &constraint.variance.coeffs)
                - ks_coeff * actual_ks_variance;
            let ks_max_variance = (constraint.safe_variance_bound - actual_variance) / ks_coeff;
            smallest_ks_max_variance = smallest_ks_max_variance.min(ks_max_variance);
        }

        smallest_ks_max_variance
    }

    pub fn fks_max_feasible_variance(
        &self,
        operations_variance: &OperationsValue,
        src_partition: usize,
        dst_partition: usize,
    ) -> f64 {
        let fks_index = operations_variance
            .index
            .keyswitch_to_big(src_partition, dst_partition);
        let actual_fks_variance = operations_variance.values[fks_index];

        let mut smallest_fks_max_variance = std::f64::MAX;

        for constraint in &self.undominated_constraints {
            let fks_coeff = constraint
                .variance
                .coeff_partition_keyswitch_to_big(src_partition, dst_partition);
            if fks_coeff == 0.0 {
                continue;
            }
            let actual_variance = f64_dot(operations_variance, &constraint.variance.coeffs)
                - fks_coeff * actual_fks_variance;
            let fks_max_variance = (constraint.safe_variance_bound - actual_variance) / fks_coeff;
            smallest_fks_max_variance = smallest_fks_max_variance.min(fks_max_variance);
        }

        smallest_fks_max_variance
    }

    pub fn feasible(&self, operations_variance: &OperationsValue) -> bool {
        if self.global_p_error.is_none() {
            self.local_feasible(operations_variance)
        } else {
            self.global_feasible(operations_variance)
        }
    }

    fn local_feasible(&self, operations_variance: &OperationsValue) -> bool {
        for constraint in &self.undominated_constraints {
            if f64_dot(operations_variance, &constraint.variance.coeffs)
                > constraint.safe_variance_bound
            {
                return false;
            };
        }
        true
    }

    fn global_feasible(&self, operations_variance: &OperationsValue) -> bool {
        self.global_p_error_with_cut(operations_variance, self.global_p_error.unwrap_or(1.0))
            .is_some()
    }

    pub fn worst_constraint(
        &self,
        operations_variance: &OperationsValue,
    ) -> (f64, f64, &VarianceConstraint) {
        let mut worst_constraint = &self.undominated_constraints[0];
        let mut worst_relative_variance = 0.0;
        let mut worst_variance = 0.0;
        for constraint in &self.undominated_constraints {
            let variance = f64_dot(operations_variance, &constraint.variance.coeffs);
            let relative_variance = variance / constraint.safe_variance_bound;
            if relative_variance > worst_relative_variance {
                worst_relative_variance = relative_variance;
                worst_variance = variance;
                worst_constraint = constraint;
            }
        }
        (worst_variance, worst_relative_variance, worst_constraint)
    }

    pub fn p_error(&self, operations_variance: &OperationsValue) -> f64 {
        let (_, relative_variance, _) = self.worst_constraint(operations_variance);
        p_error_from_relative_variance(relative_variance, self.kappa)
    }

    fn global_p_error_with_cut(
        &self,
        operations_variance: &OperationsValue,
        cut: f64,
    ) -> Option<f64> {
        let mut global_p_error = 0.0;
        for constraint in &self.constraints {
            let variance = f64_dot(operations_variance, &constraint.variance.coeffs);
            let relative_variance = variance / constraint.safe_variance_bound;
            let p_error = p_error_from_relative_variance(relative_variance, self.kappa);
            global_p_error = combine_errors(
                global_p_error,
                repeat_p_error(p_error, constraint.nb_constraints),
            );
            if global_p_error > cut {
                return None;
            }
        }
        Some(global_p_error)
    }

    pub fn global_p_error(&self, operations_variance: &OperationsValue) -> f64 {
        self.global_p_error_with_cut(operations_variance, 1.0)
            .unwrap_or(1.0)
    }

    pub fn filter_constraints(&self, partition: PartitionIndex) -> Self {
        let nb_partitions = self.constraints[0].variance.nb_partitions();
        let touch_any_ks = |constraint: &VarianceConstraint, i| {
            let variance = &constraint.variance;
            variance.coeff_keyswitch_to_small(partition, i) > 0.0
                || variance.coeff_keyswitch_to_small(i, partition) > 0.0
                || variance.coeff_partition_keyswitch_to_big(partition, i) > 0.0
                || variance.coeff_partition_keyswitch_to_big(i, partition) > 0.0
        };
        let partition_constraints: Vec<_> = self
            .constraints
            .iter()
            .filter(|constraint| {
                constraint.partition == partition
                    || (0..nb_partitions).any(|i| touch_any_ks(constraint, i))
            })
            .map(VarianceConstraint::clone)
            .collect();
        Self::of(&partition_constraints, self.kappa, self.global_p_error)
    }

    pub fn compressed(self) -> Self {
        let mut detect_used: Vec<bool> = vec![false; self.constraints[0].variance.coeffs.len()];
        for constraint in &self.constraints {
            for (i, &coeff) in constraint.variance.coeffs.iter().enumerate() {
                if coeff > 0.0 {
                    detect_used[i] = true;
                }
            }
        }
        let compress = |c: &VarianceConstraint| VarianceConstraint {
            variance: c.variance.compress(&detect_used),
            ..(*c)
        };
        let constraints = self.constraints.iter().map(compress).collect();
        let undominated_constraints = self.undominated_constraints.iter().map(compress).collect();
        Self {
            constraints,
            undominated_constraints,
            ..self
        }
    }

    pub fn zero_variance(&self) -> OperationsValue {
        if self.undominated_constraints[0]
            .variance
            .coeffs
            .index
            .is_compressed()
        {
            OperationsValue::zero_compressed(&self.undominated_constraints[0].variance.coeffs.index)
        } else {
            OperationsValue::zero(
                self.undominated_constraints[0]
                    .variance
                    .coeffs
                    .nb_partitions(),
            )
        }
    }
}
