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
        let index = counts.index;
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
}
