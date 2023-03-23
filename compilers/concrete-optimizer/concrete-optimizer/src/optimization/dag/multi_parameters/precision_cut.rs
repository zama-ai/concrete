use crate::dag::operator::{Operator, Precision};
use crate::dag::unparametrized;
use crate::optimization::dag::multi_parameters::partitions::PartitionIndex;

#[derive(Clone, Debug)]
pub struct PrecisionCut {
    // partition0 precision <= p_cut[0] < partition 1 precision <= p_cut[1] ...
    // precision are in the sens of Lut input precision and are sorted
    pub p_cut: Vec<Precision>,
}

impl PrecisionCut {
    pub fn partition(
        &self,
        dag: &unparametrized::OperationDag,
        op: &Operator,
    ) -> Option<PartitionIndex> {
        match op {
            Operator::Lut { input, .. } => {
                assert!(!self.p_cut.is_empty());
                for (partition, &p_cut) in self.p_cut.iter().enumerate() {
                    if dag.out_precisions[input.i] <= p_cut {
                        return Some(partition);
                    }
                }
                Some(self.p_cut.len())
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for PrecisionCut {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut prev_p_cut = 0;
        for (partition, &p_cut) in self.p_cut.iter().enumerate() {
            writeln!(
                f,
                "partition {partition}: {prev_p_cut} up through {p_cut} bits"
            )?;
            prev_p_cut = p_cut + 1;
        }
        writeln!(
            f,
            "partition {}: {prev_p_cut} bits and higher",
            self.p_cut.len()
        )?;
        Ok(())
    }
}
