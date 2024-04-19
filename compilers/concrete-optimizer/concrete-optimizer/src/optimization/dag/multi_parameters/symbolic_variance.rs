use std::fmt;

use crate::optimization::dag::multi_parameters::operations_value::OperationsValue;

use super::partitions::PartitionIndex;

/**
 * A variance that is represented as a linear combination of base variances.
 * Only the linear coefficient are known.
 * The base variances are unknown.
 *
 * Possible base variances:
 *  - fresh,
 *  - lut output,
 *  - keyswitch,
 *  - partition keyswitch,
 *  - modulus switching
 *
 * We only known that the fresh <= lut output in the same partition.
 * Each linear coefficient is a variance factor.
 * There are homogeneous to squared weight (or summed square weights or squared norm2).
 */
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SymbolicVariance {
    pub partition: PartitionIndex,
    pub coeffs: OperationsValue,
}

impl SymbolicVariance {
    // To be used as a initial accumulator
    pub const ZERO: Self = Self {
        partition: PartitionIndex::FIRST,
        coeffs: OperationsValue::ZERO,
    };

    pub fn nb_partitions(&self) -> usize {
        self.coeffs.nb_partitions()
    }

    pub fn nan(nb_partitions: usize) -> Self {
        Self {
            partition: PartitionIndex::INVALID,
            coeffs: OperationsValue::nan(nb_partitions),
        }
    }

    pub fn input(nb_partitions: usize, partition: PartitionIndex) -> Self {
        let mut r = Self {
            partition,
            coeffs: OperationsValue::zero(nb_partitions),
        };
        // rust ..., offset cannot be inlined
        *r.coeffs.input(partition) = 1.0;
        r
    }

    pub fn coeff_input(&self, partition: PartitionIndex) -> f64 {
        self.coeffs[self.coeffs.index.input(partition)]
    }

    pub fn after_pbs(nb_partitions: usize, partition: PartitionIndex) -> Self {
        let mut r = Self {
            partition,
            coeffs: OperationsValue::zero(nb_partitions),
        };
        *r.coeffs.pbs(partition) = 1.0;
        r
    }

    pub fn coeff_pbs(&self, partition: PartitionIndex) -> f64 {
        self.coeffs[self.coeffs.index.pbs(partition)]
    }

    pub fn coeff_modulus_switching(&self, partition: PartitionIndex) -> f64 {
        self.coeffs[self.coeffs.index.modulus_switching(partition)]
    }

    pub fn after_modulus_switching(&self, partition: PartitionIndex) -> Self {
        let mut new = self.clone();
        let index = self.coeffs.index.modulus_switching(partition);
        assert!(new.coeffs[index] == 0.0);
        new.coeffs[index] = 1.0;
        new
    }

    pub fn coeff_keyswitch_to_small(
        &self,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> f64 {
        self.coeffs[self
            .coeffs
            .index
            .keyswitch_to_small(src_partition, dst_partition)]
    }

    pub fn after_partition_keyswitch_to_small(
        &self,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> Self {
        let index = self
            .coeffs
            .index
            .keyswitch_to_small(src_partition, dst_partition);
        self.after_partition_keyswitch(src_partition, dst_partition, index)
    }

    pub fn coeff_partition_keyswitch_to_big(
        &self,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> f64 {
        self.coeffs[self
            .coeffs
            .index
            .keyswitch_to_big(src_partition, dst_partition)]
    }

    pub fn after_partition_keyswitch_to_big(
        &self,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
    ) -> Self {
        let index = self
            .coeffs
            .index
            .keyswitch_to_big(src_partition, dst_partition);
        self.after_partition_keyswitch(src_partition, dst_partition, index)
    }

    pub fn after_partition_keyswitch(
        &self,
        src_partition: PartitionIndex,
        dst_partition: PartitionIndex,
        index: usize,
    ) -> Self {
        assert!(src_partition.0 < self.nb_partitions());
        assert!(dst_partition.0 < self.nb_partitions());
        assert!(src_partition == self.partition);
        let mut new = self.clone();
        new.partition = dst_partition;
        new.coeffs[index] = 1.0;
        new
    }

    #[allow(clippy::float_cmp)]
    pub fn after_levelled_op(&self, manp: f64) -> Self {
        let new_coeff = manp * manp;
        // detect the previous base manp level
        // this is the maximum value of fresh base noise and pbs base noise
        let mut current_max: f64 = 0.0;
        for partition in PartitionIndex::range(0, self.nb_partitions()) {
            let fresh_coeff = self.coeff_input(partition);
            let pbs_noise_coeff = self.coeff_pbs(partition);
            current_max = current_max.max(fresh_coeff).max(pbs_noise_coeff);
        }
        // assert!(1.0 <= current_max);
        // assert!(
        //     current_max <= new_coeff,
        //     "Non monotonious levelled op: {current_max} <= {new_coeff}"
        // );
        // replace all current_max by new_coeff
        // multiply everything else by new_coeff / current_max
        let mut new = self.clone();
        if current_max == 0.0 {
            return new;
        }
        for cell in &mut new.coeffs.values {
            if *cell == current_max {
                *cell = new_coeff;
            } else {
                *cell *= new_coeff / current_max;
            }
        }
        new
    }

    pub fn max(&self, other: &Self) -> Self {
        let mut coeffs = self.coeffs.clone();
        for (i, coeff) in coeffs.iter_mut().enumerate() {
            *coeff = coeff.max(other.coeffs[i]);
        }
        Self { coeffs, ..*self }
    }

    pub fn compress(&self, detect_used: &[bool]) -> Self {
        Self {
            coeffs: self.coeffs.compress(detect_used),
            ..(*self)
        }
    }
}

impl fmt::Display for SymbolicVariance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self == &Self::ZERO {
            return write!(f, "ZERO x σ²");
        }
        if self.coeffs[0].is_nan() {
            return write!(f, "NAN x σ²");
        }
        let mut add_plus = "";
        for src_partition in PartitionIndex::range(0, self.nb_partitions()) {
            let coeff = self.coeff_input(src_partition);
            if coeff != 0.0 {
                write!(f, "{add_plus}{coeff}σ²In[{src_partition}]")?;
                add_plus = " + ";
            }
            let coeff = self.coeff_pbs(src_partition);
            if coeff != 0.0 {
                write!(f, "{add_plus}{coeff}σ²Br[{src_partition}]")?;
                add_plus = " + ";
            }
            for dst_partition in PartitionIndex::range(0, self.nb_partitions()) {
                let coeff = self.coeff_partition_keyswitch_to_big(src_partition, dst_partition);
                if coeff != 0.0 {
                    write!(f, "{add_plus}{coeff}σ²FK[{src_partition}→{dst_partition}]")?;
                    add_plus = " + ";
                }
            }
        }
        for src_partition in PartitionIndex::range(0, self.nb_partitions()) {
            for dst_partition in PartitionIndex::range(0, self.nb_partitions()) {
                let coeff = self.coeff_keyswitch_to_small(src_partition, dst_partition);
                if coeff != 0.0 {
                    if src_partition == dst_partition {
                        write!(f, "{add_plus}{coeff}σ²K[{src_partition}]")?;
                    } else {
                        write!(f, "{add_plus}{coeff}σ²K[{src_partition}→{dst_partition}]")?;
                    }
                    add_plus = " + ";
                }
            }
        }
        for partition in PartitionIndex::range(0, self.nb_partitions()) {
            let coeff = self.coeff_modulus_switching(partition);
            if coeff != 0.0 {
                write!(f, "{add_plus}{coeff}σ²M[{partition}]")?;
                add_plus = " + ";
            }
        }
        Ok(())
    }
}

impl std::ops::Add for SymbolicVariance {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        if self.coeffs.is_empty() {
            self = rhs;
        } else {
            for i in 0..self.coeffs.len() {
                self.coeffs[i] += rhs.coeffs[i];
            }
        };
        self
    }
}

impl std::ops::AddAssign for SymbolicVariance {
    fn add_assign(&mut self, rhs: Self) {
        if self.coeffs.is_empty() {
            *self = rhs;
        } else {
            for i in 0..self.coeffs.len() {
                self.coeffs[i] += rhs.coeffs[i];
            }
        }
    }
}

impl std::ops::Mul<f64> for SymbolicVariance {
    type Output = Self;
    fn mul(self, sq_weight: f64) -> Self {
        Self {
            coeffs: self.coeffs * sq_weight,
            ..self
        }
    }
}
