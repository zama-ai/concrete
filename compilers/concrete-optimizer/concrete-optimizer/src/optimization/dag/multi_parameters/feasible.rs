use crate::noise_estimator::p_error::{combine_errors, repeat_p_error};
use crate::optimization::dag::multi_parameters::variance_constraint::VarianceConstraint;
use crate::optimization::dag::solo_key::analyze::p_error_from_relative_variance;
use crate::utils::f64::f64_dot;

use super::operations_value::OperationsValue;
use super::partitions::PartitionIndex;

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
}
