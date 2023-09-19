use std::fmt;

use crate::utils::f64::f64_dot;

use super::operations_value::OperationsValue;

#[derive(Clone, Debug)]
pub struct OperationsCount {
    pub counts: OperationsValue,
}

#[derive(Clone, Debug)]
pub struct OperationsCost {
    pub costs: OperationsValue,
}

#[derive(Clone, Debug)]
pub struct Complexity {
    pub counts: OperationsValue,
}

impl fmt::Display for OperationsCount {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut add_plus = "";
        let counts = &self.counts;
        let nb_partitions = counts.nb_partitions();
        let index = &counts.index;
        for src_partition in 0..nb_partitions {
            for dst_partition in 0..nb_partitions {
                let coeff = counts.values[index.keyswitch_to_small(src_partition, dst_partition)];
                if coeff != 0.0 {
                    if src_partition == dst_partition {
                        write!(f, "{add_plus}{coeff}¢K[{src_partition}]")?;
                    } else {
                        write!(f, "{add_plus}{coeff}¢K[{src_partition}→{dst_partition}]")?;
                    }
                    add_plus = " + ";
                }
            }
        }
        for src_partition in 0..nb_partitions {
            assert!(counts.values[index.input(src_partition)] == 0.0);
            let coeff = counts.values[index.pbs(src_partition)];
            if coeff != 0.0 {
                write!(f, "{add_plus}{coeff}¢Br[{src_partition}]")?;
                add_plus = " + ";
            }
            for dst_partition in 0..nb_partitions {
                let coeff = counts.values[index.keyswitch_to_big(src_partition, dst_partition)];
                if coeff != 0.0 {
                    write!(f, "{add_plus}{coeff}¢FK[{src_partition}→{dst_partition}]")?;
                    add_plus = " + ";
                }
            }
        }

        for partition in 0..nb_partitions {
            assert!(counts.values[index.modulus_switching(partition)] == 0.0);
        }
        if add_plus.is_empty() {
            write!(f, "ZERO x ¢")?;
        }
        Ok(())
    }
}

impl Complexity {
    pub fn of(counts: &OperationsCount) -> Self {
        Self {
            counts: counts.counts.clone(),
        }
    }

    pub fn complexity(&self, costs: &OperationsValue) -> f64 {
        f64_dot(&self.counts, costs)
    }

    pub fn ks_max_cost(
        &self,
        complexity_cut: f64,
        costs: &OperationsValue,
        src_partition: usize,
        dst_partition: usize,
    ) -> f64 {
        let ks_index = costs.index.keyswitch_to_small(src_partition, dst_partition);
        let actual_ks_cost = costs.values[ks_index];
        let ks_coeff = self.counts[self
            .counts
            .index
            .keyswitch_to_small(src_partition, dst_partition)];
        let actual_complexity = self.complexity(costs) - ks_coeff * actual_ks_cost;

        (complexity_cut - actual_complexity) / ks_coeff
    }

    pub fn fks_max_cost(
        &self,
        complexity_cut: f64,
        costs: &OperationsValue,
        src_partition: usize,
        dst_partition: usize,
    ) -> f64 {
        let fks_index = costs.index.keyswitch_to_big(src_partition, dst_partition);
        let actual_fks_cost = costs.values[fks_index];
        let fks_coeff = self.counts[self
            .counts
            .index
            .keyswitch_to_big(src_partition, dst_partition)];
        let actual_complexity = self.complexity(costs) - fks_coeff * actual_fks_cost;

        (complexity_cut - actual_complexity) / fks_coeff
    }

    pub fn compressed(self) -> Self {
        let mut detect_used: Vec<bool> = vec![false; self.counts.len()];
        for (i, &count) in self.counts.iter().enumerate() {
            if count > 0.0 {
                detect_used[i] = true;
            }
        }
        Self {
            counts: self.counts.compress(&detect_used),
        }
    }

    pub fn zero_cost(&self) -> OperationsValue {
        if self.counts.index.is_compressed() {
            OperationsValue::zero_compressed(&self.counts.index)
        } else {
            OperationsValue::zero(self.counts.nb_partitions())
        }
    }
}
