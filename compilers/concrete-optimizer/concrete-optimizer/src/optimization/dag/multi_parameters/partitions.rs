use std::collections::HashSet;

pub type PartitionIndex = usize;
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
