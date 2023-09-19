use std::ops::{Deref, DerefMut};

/**
 * Index actual operations (input, ks, pbs, fks, modulus switching, etc).
 */
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub struct Indexing {
    /* Values order
    [
        // Partition 1
        // related only to the partition
        fresh, pbs, modulus,
        // Keyswitchs to small, from any partition to 1
        ks from 1, ks from 2, ...
        // Keyswitch to big, from any partition to 1
        ks from 1, ks from 2, ...

        // Partition 2
        // same
    ]
    */
    pub nb_partitions: usize,
}

pub const VALUE_INDEX_FRESH: usize = 0;
pub const VALUE_INDEX_PBS: usize = 1;
pub const VALUE_INDEX_MODULUS: usize = 2;
// number of value always present for a partition
pub const STABLE_NB_VALUES_BY_PARTITION: usize = 3;

impl Indexing {
    fn nb_keyswitchs_per_partition(self) -> usize {
        self.nb_partitions
    }

    pub fn nb_coeff_per_partition(self) -> usize {
        STABLE_NB_VALUES_BY_PARTITION + 2 * self.nb_partitions
    }

    pub fn nb_coeff(self) -> usize {
        self.nb_partitions * (STABLE_NB_VALUES_BY_PARTITION + 2 * self.nb_partitions)
    }

    pub fn input(self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        partition * self.nb_coeff_per_partition() + VALUE_INDEX_FRESH
    }

    pub fn pbs(self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        partition * self.nb_coeff_per_partition() + VALUE_INDEX_PBS
    }

    pub fn modulus_switching(self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        partition * self.nb_coeff_per_partition() + VALUE_INDEX_MODULUS
    }

    pub fn keyswitch_to_small(self, src_partition: usize, dst_partition: usize) -> usize {
        assert!(src_partition < self.nb_partitions);
        assert!(dst_partition < self.nb_partitions);
        // Skip other partition
        dst_partition * self.nb_coeff_per_partition()
        // Skip non keyswitchs
        + STABLE_NB_VALUES_BY_PARTITION
        // Select the right keyswicth to small
        + src_partition
    }

    pub fn keyswitch_to_big(self, src_partition: usize, dst_partition: usize) -> usize {
        assert!(src_partition < self.nb_partitions);
        assert!(dst_partition < self.nb_partitions);
        // Skip other partition
        dst_partition * self.nb_coeff_per_partition()
        // Skip non keyswitchs
        + STABLE_NB_VALUES_BY_PARTITION
        // Skip keyswitch to small
        + self.nb_keyswitchs_per_partition()
        // Select the right keyswicth to big
        + src_partition
    }
}

/**
 * Represent any values indexed by actual operations (input, pbs, modulus switching, ks, fks, , etc) variance,
 */
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct OperationsValue {
    pub index: Indexing,
    pub values: Vec<f64>,
}

impl OperationsValue {
    pub const ZERO: Self = Self {
        index: Indexing { nb_partitions: 0 },
        values: vec![],
    };

    pub fn zero(nb_partitions: usize) -> Self {
        let index = Indexing { nb_partitions };
        Self {
            index,
            values: vec![0.0; index.nb_coeff()],
        }
    }

    pub fn nan(nb_partitions: usize) -> Self {
        let index = Indexing { nb_partitions };
        Self {
            index,
            values: vec![f64::NAN; index.nb_coeff()],
        }
    }

    pub fn input(&mut self, partition: usize) -> &mut f64 {
        &mut self.values[self.index.input(partition)]
    }

    pub fn pbs(&mut self, partition: usize) -> &mut f64 {
        &mut self.values[self.index.pbs(partition)]
    }

    pub fn ks(&mut self, src_partition: usize, dst_partition: usize) -> &mut f64 {
        &mut self.values[self.index.keyswitch_to_small(src_partition, dst_partition)]
    }

    pub fn fks(&mut self, src_partition: usize, dst_partition: usize) -> &mut f64 {
        &mut self.values[self.index.keyswitch_to_big(src_partition, dst_partition)]
    }

    pub fn modulus_switching(&mut self, partition: usize) -> &mut f64 {
        &mut self.values[self.index.modulus_switching(partition)]
    }

    pub fn nb_partitions(&self) -> usize {
        self.index.nb_partitions
    }
}

impl Deref for OperationsValue {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl DerefMut for OperationsValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl std::ops::AddAssign for OperationsValue {
    fn add_assign(&mut self, rhs: Self) {
        if self.values.is_empty() {
            *self = rhs;
        } else {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i];
            }
        }
    }
}

impl std::ops::AddAssign<&Self> for OperationsValue {
    fn add_assign(&mut self, rhs: &Self) {
        if self.values.is_empty() {
            *self = rhs.clone();
        } else {
            for i in 0..self.values.len() {
                self.values[i] += rhs.values[i];
            }
        }
    }
}

impl std::ops::Mul<f64> for OperationsValue {
    type Output = Self;
    fn mul(self, sq_weight: f64) -> Self {
        Self {
            values: self.values.iter().map(|v| v * sq_weight).collect(),
            ..self
        }
    }
}
