use std::{
    collections::HashSet,
    fmt::Display,
    ops::{Deref, Index, IndexMut},
};

use crate::dag::operator::OperatorIndex;

#[derive(Clone, Debug, PartialEq, Eq, Default, PartialOrd, Ord, Hash, Copy)]
pub struct PartitionIndex(pub(crate) usize);

impl PartitionIndex {
    pub const FIRST: Self = Self(0);

    pub const INVALID: Self = Self(usize::MAX);

    pub fn range(from: usize, to: usize) -> impl DoubleEndedIterator<Item = Self> {
        (from..to).map(PartitionIndex)
    }
}

impl Display for PartitionIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Deref for PartitionIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type AdditionalRepresentations = HashSet<PartitionIndex>;

// How one input is made compatible with the instruction partition
#[derive(Clone, Debug, PartialEq, Eq)]

pub enum Transition {
    // The input rely on an already converted input, multi representation value
    Additional { src_partition: PartitionIndex },
    // The input can be converted directly by the internal instructions keyswitch
    Internal { src_partition: PartitionIndex },
}

// One instruction partition is computed for each instruction.
// It represents its partition and relations with other partitions.
#[derive(Clone, Debug, Default)]
pub struct InstructionPartition {
    // The partition assigned to the instruction
    #[allow(unknown_lints)]
    #[allow(clippy::struct_field_names)]
    pub instruction_partition: PartitionIndex,
    // How the input are made compatible with the instruction partition
    pub inputs_transition: Vec<Option<Transition>>,
    // How the output are made compatible with levelled operation
    pub alternative_output_representation: AdditionalRepresentations,
}

impl InstructionPartition {
    pub fn new(instruction_partition: PartitionIndex) -> Self {
        Self {
            instruction_partition,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub fn no_transition(&self) -> bool {
        self.alternative_output_representation.is_empty()
            && self.inputs_transition.iter().all(Option::is_none)
    }
}

#[derive(Clone, Debug)]
pub struct Partitions {
    pub nb_partitions: usize,
    pub instrs_partition: Vec<InstructionPartition>,
}

impl Index<OperatorIndex> for Partitions {
    type Output = InstructionPartition;

    fn index(&self, index: OperatorIndex) -> &Self::Output {
        &self.instrs_partition[index.0]
    }
}

impl IndexMut<OperatorIndex> for Partitions {
    fn index_mut(&mut self, index: OperatorIndex) -> &mut Self::Output {
        &mut self.instrs_partition[index.0]
    }
}

#[allow(unused)]
pub struct PartitionsCircuit<'part> {
    pub(crate) partitions: &'part [InstructionPartition],
    pub(crate) idx: Vec<usize>,
}

impl<'part> PartitionsCircuit<'part> {
    #[allow(unused)]
    pub fn get_node_iter(&'part self) -> impl Iterator<Item = &'part InstructionPartition> {
        self.idx.iter().map(|i| self.partitions.get(*i).unwrap())
    }
}
