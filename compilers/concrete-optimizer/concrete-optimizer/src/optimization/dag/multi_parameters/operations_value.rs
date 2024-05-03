use std::ops::{Deref, DerefMut};

/**
 * Index actual operations (input, ks, pbs, fks, modulus switching, etc).
 */
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
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
    pub compressed_index: Vec<usize>,
}

const VALUE_INDEX_FRESH: usize = 0;
const VALUE_INDEX_PBS: usize = 1;
const VALUE_INDEX_MODULUS: usize = 2;
const VALUE_INDEX_FRESH_PUBLIC: usize = 3;
// number of value always present for a partition
const STABLE_NB_VALUES_BY_PARTITION: usize = 4;

pub const COMPRESSED_0_INDEX: usize = 0; // all 0.0 value are indexed here
pub const COMPRESSED_FIRST_FREE_INDEX: usize = 1;

impl Indexing {
    fn uncompressed(nb_partitions: usize) -> Self {
        Self {
            nb_partitions,
            compressed_index: vec![],
        }
    }

    fn compress(&self, used: &[bool]) -> Self {
        assert!(!self.is_compressed());
        let mut compressed_index = vec![COMPRESSED_0_INDEX; self.nb_coeff()];
        let mut index = COMPRESSED_FIRST_FREE_INDEX;
        for (i, &is_used) in used.iter().enumerate() {
            if is_used {
                compressed_index[i] = index;
                index += 1;
            }
        }
        Self {
            compressed_index,
            ..(*self)
        }
    }

    pub fn is_compressed(&self) -> bool {
        !self.compressed_index.is_empty()
    }

    fn nb_keyswitchs_per_partition(&self) -> usize {
        self.nb_partitions
    }

    pub fn maybe_compressed(&self, i: usize) -> usize {
        if self.is_compressed() {
            self.compressed_index[i]
        } else {
            i
        }
    }

    pub fn nb_coeff_per_partition(&self) -> usize {
        STABLE_NB_VALUES_BY_PARTITION + 2 * self.nb_partitions
    }

    pub fn nb_coeff(&self) -> usize {
        self.nb_partitions * (STABLE_NB_VALUES_BY_PARTITION + 2 * self.nb_partitions)
    }

    pub fn input(&self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        self.maybe_compressed(partition * self.nb_coeff_per_partition() + VALUE_INDEX_FRESH)
    }

    pub fn public_input(&self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        self.maybe_compressed(partition * self.nb_coeff_per_partition() + VALUE_INDEX_FRESH_PUBLIC)
    }

    pub fn pbs(&self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        self.maybe_compressed(partition * self.nb_coeff_per_partition() + VALUE_INDEX_PBS)
    }

    pub fn modulus_switching(&self, partition: usize) -> usize {
        assert!(partition < self.nb_partitions);
        self.maybe_compressed(partition * self.nb_coeff_per_partition() + VALUE_INDEX_MODULUS)
    }

    pub fn keyswitch_to_small(&self, src_partition: usize, dst_partition: usize) -> usize {
        assert!(src_partition < self.nb_partitions);
        assert!(dst_partition < self.nb_partitions);
        self.maybe_compressed(
            // Skip other partition
            dst_partition * self.nb_coeff_per_partition()
            // Skip non keyswitchs
            + STABLE_NB_VALUES_BY_PARTITION
            // Select the right keyswicth to small
            + src_partition,
        )
    }

    pub fn keyswitch_to_big(&self, src_partition: usize, dst_partition: usize) -> usize {
        assert!(src_partition < self.nb_partitions);
        assert!(dst_partition < self.nb_partitions);
        self.maybe_compressed(
            // Skip other partition
            dst_partition * self.nb_coeff_per_partition()
            // Skip non keyswitchs
            + STABLE_NB_VALUES_BY_PARTITION
            // Skip keyswitch to small
            + self.nb_keyswitchs_per_partition()
            // Select the right keyswicth to big
            + src_partition,
        )
    }

    pub fn compressed_size(&self) -> usize {
        self.compressed_index.iter().copied().max().unwrap_or(0) + 1
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
        index: Indexing {
            nb_partitions: 0,
            compressed_index: vec![],
        },
        values: vec![],
    };

    pub fn zero(nb_partitions: usize) -> Self {
        let index = Indexing::uncompressed(nb_partitions);
        let nb_coeff = index.nb_coeff();
        Self {
            index,
            values: vec![0.0; nb_coeff],
        }
    }

    pub fn zero_compressed(index: &Indexing) -> Self {
        assert!(index.is_compressed());
        Self {
            index: index.clone(),
            values: vec![0.0; index.compressed_size()],
        }
    }

    pub fn nan(nb_partitions: usize) -> Self {
        let index = Indexing::uncompressed(nb_partitions);
        let nb_coeff = index.nb_coeff();
        Self {
            index,
            values: vec![f64::NAN; nb_coeff],
        }
    }

    pub fn input(&mut self, partition: usize) -> &mut f64 {
        &mut self.values[self.index.input(partition)]
    }

    pub fn public_input(&mut self, partition: usize) -> &mut f64 {
        &mut self.values[self.index.public_input(partition)]
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

    pub fn compress(&self, used: &[bool]) -> Self {
        self.compress_with(self.index.compress(used))
    }

    pub fn compress_like(&self, other: Self) -> Self {
        self.compress_with(other.index)
    }

    fn compress_with(&self, index: Indexing) -> Self {
        assert!(!index.compressed_index.is_empty());
        assert!(self.index.compressed_index.is_empty());
        let mut values = vec![0.0; index.compressed_size()];
        for (i, &value) in self.values.iter().enumerate() {
            #[allow(clippy::option_if_let_else)]
            let j = index.compressed_index[i];
            if j == COMPRESSED_0_INDEX {
                assert!(value == 0.0, "Cannot compress non null value");
            } else {
                values[j] = value;
            }
        }
        assert!(values[COMPRESSED_0_INDEX] == 0.0);
        assert!(!index.compressed_index.is_empty());
        Self { index, values }
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
