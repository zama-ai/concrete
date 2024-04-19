use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;

use crate::dag::operator::{Operator, OperatorIndex, Precision};
use crate::dag::rewrite::round::expand_round_and_index_map;
use crate::dag::unparametrized;
use crate::optimization::dag::multi_parameters::partitions::PartitionIndex;
use crate::optimization::dag::solo_key::analyze::out_variances;
use crate::optimization::dag::solo_key::symbolic_variance::SymbolicVariance;

const ROUND_INNER_MULTI_PARAMETER: bool = false;
const ROUND_EXTERNAL_MULTI_PARAMETER: bool = !ROUND_INNER_MULTI_PARAMETER && true;

// TODO: keep both precisions
// TODO: rounding lut should have its own partition based on max norm2 and precisions
#[derive(Clone, Debug)]
pub struct PartitionCut {
    // partition0 precision <= p_cut[0] < partition 1 precision <= p_cut[1] ...
    // precision are in the sens of Lut input precision and are sorted
    pub p_cut: Vec<(Precision, f64)>,

    // # TODO RELATIVE NORM2
    // # HIGHER NORM2 MEANS HIGHER VARIANCE IN CONSTRAINT
    // norm2 * 2 ** (out precision - in precision)
    pub rnorm2: Vec<f64>,
}

impl PartitionCut {
    pub fn empty() -> Self {
        Self {
            p_cut: vec![],
            rnorm2: vec![],
        }
    }

    pub fn from_precisions(precisions: &[Precision]) -> Self {
        let mut precisions: Vec<_> = precisions.to_vec();
        precisions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        _ = precisions.pop();

        Self {
            p_cut: precisions.iter().map(|p| (*p, f64::MAX)).collect(),
            rnorm2: vec![],
        }
    }

    fn rnorm2(&self, op_i: OperatorIndex) -> f64 {
        if self.rnorm2.is_empty() {
            return f64::MAX;
        }
        assert!(!self.rnorm2[op_i.0].is_nan());
        self.rnorm2[op_i.0]
    }

    pub fn partition(
        &self,
        dag: &unparametrized::Dag,
        op_i: OperatorIndex,
    ) -> Option<PartitionIndex> {
        let op = &dag.operators[op_i.0];
        match op {
            Operator::Lut { input, .. } => {
                assert!(!self.p_cut.is_empty());
                for (partition, &(precision_cut, norm2_cut)) in self.p_cut.iter().enumerate() {
                    if dag.out_precisions[input.0] <= precision_cut
                        && self.rnorm2(op_i) <= norm2_cut
                    {
                        return Some(PartitionIndex(partition));
                    }
                }
                Some(PartitionIndex(self.p_cut.len()))
            }
            _ => None,
        }
    }

    pub fn for_each_precision(dag: &unparametrized::Dag) -> Self {
        let (dag, _) = expand_round_and_index_map(dag);
        let mut lut_in_precisions: HashSet<_> = HashSet::default();
        for op in &dag.operators {
            if let Operator::Lut { input, .. } = op {
                _ = lut_in_precisions.insert(dag.out_precisions[input.0]);
            }
        }
        let precisions: Vec<_> = lut_in_precisions.iter().copied().collect();
        Self::from_precisions(&precisions)
    }

    pub fn maximal_partitionning(original_dag: &unparametrized::Dag) -> Self {
        // Note: only keep one 0-bits, partition as the compiler will not support multi-parameter round
        // partition based on input precision and output log norm2
        let (dag, rewrited) = expand_round_and_index_map(original_dag);
        let mut round_index: HashMap<usize, usize> = HashMap::default();
        for (round_i, op) in original_dag.operators.iter().enumerate() {
            if let Operator::Round { .. } = op {
                for op in &rewrited[round_i] {
                    let already = round_index.insert(op.0, round_i);
                    assert!(already.is_none());
                }
            }
        }
        let out_variances: Vec<SymbolicVariance> = out_variances(&dag);
        let mut noise_origins: Vec<HashSet<usize>> = vec![HashSet::default(); out_variances.len()];
        let mut max_output_norm2 = vec![f64::NAN; out_variances.len()];

        assert!(out_variances.len() == dag.operators.len());
        // Find input lut log norm2 and lut as origins
        for (op_i, op) in dag.operators.iter().enumerate() {
            match op {
                // propagate
                Operator::Dot { inputs, .. } | Operator::LevelledOp { inputs, .. } => {
                    let mut origins = HashSet::default();
                    for input in inputs {
                        origins.extend(&noise_origins[input.0]);
                    }
                    noise_origins[op_i] = origins;
                }
                #[allow(clippy::assigning_clones)]
                Operator::UnsafeCast { input, .. } => {
                    noise_origins[op_i] = noise_origins[input.0].clone();
                }
                // origins
                Operator::Lut { .. } => {
                    noise_origins[op_i] = std::iter::once(op_i).collect();
                }
                Operator::Input { .. } => {
                    max_output_norm2[op_i] = 1.0; // initial value that can be maxed
                    noise_origins[op_i] = std::iter::once(op_i).collect();
                }
                // unreachable
                Operator::Round { .. } => panic!("expand_round failed"),
            }
        }
        let out_norm2 = |i: usize| out_variances[i].lut_coeff + out_variances[i].input_coeff;
        // convert input_norm2 to max_output_norm2
        let mut lut_partition: HashSet<_> = HashSet::default();
        for dest in &dag.operators {
            if let Operator::Lut { input, .. } = dest {
                for &origin in &noise_origins[input.0] {
                    let norm2 = out_norm2(input.0);
                    max_output_norm2[origin] = max_output_norm2[origin].max(norm2);
                    assert!(!max_output_norm2[origin].is_nan());
                }
            }
        }
        for op in dag.get_output_operators_iter() {
            for &origin in &noise_origins[op.id.0] {
                max_output_norm2[origin] = max_output_norm2[origin].max(out_norm2(op.id.0));
                assert!(!max_output_norm2[origin].is_nan());
            }
        }
        let mut round_done: HashMap<usize, u64> = HashMap::default();
        // reassociate all lut's output_norm2 and precisions
        for (op_i, output_norm2) in max_output_norm2.iter_mut().enumerate() {
            if let Operator::Lut { input, .. } = dag.operators[op_i] {
                let input_precision = dag.out_precisions[input.0];
                let output_precision = dag.out_precisions[op_i] as i32;
                let delta_precision = output_precision - input_precision as i32;
                assert!(!output_norm2.is_nan());
                let original_norm2 = output_norm2.sqrt();
                let quantization = 1.0;
                let mut lnorm2 = (original_norm2 * 2.0_f64.powi(delta_precision)).log2();
                #[allow(clippy::cast_sign_loss)]
                if ROUND_INNER_MULTI_PARAMETER || !round_index.contains_key(&op_i) {
                    lnorm2 = (lnorm2 / quantization).ceil() * quantization;
                    *output_norm2 = lnorm2;
                } else if ROUND_EXTERNAL_MULTI_PARAMETER {
                    lnorm2 = lnorm2.ceil();
                    let round_i = round_index[&op_i];
                    if round_done.contains_key(&round_i) {
                        *output_norm2 = round_done[&round_i] as f64;
                        continue;
                    }
                    *output_norm2 = lnorm2;
                    _ = round_done.insert(round_i, *output_norm2 as u64);
                } else {
                    *output_norm2 = f64::INFINITY;
                }
                _ = lut_partition.insert((input_precision, OrderedFloat(*output_norm2)));
            }
        }
        let mut p_cut: Vec<_> = lut_partition.iter().copied().collect();
        p_cut.sort_by(|a, b| a.partial_cmp(b).unwrap());
        _ = p_cut.pop();
        let p_cut = p_cut.iter().map(|(p, n)| (*p, n.into_inner())).collect();
        Self {
            p_cut,
            rnorm2: max_output_norm2,
        }
    }

    pub fn delete_unused_cut(&self, used: &HashSet<PartitionIndex>) -> Self {
        let mut p_cut = vec![];
        for (i, &cut) in self.p_cut.iter().enumerate() {
            if used.contains(&PartitionIndex(i)) {
                p_cut.push(cut);
            }
        }
        Self {
            p_cut,
            rnorm2: self.rnorm2.clone(),
        }
    }
}

impl std::fmt::Display for PartitionCut {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut prev_precision_cut = 0;
        for (partition, &(precision_cut, norm2_cut)) in self.p_cut.iter().enumerate() {
            prev_precision_cut = prev_precision_cut.min(precision_cut);
            if prev_precision_cut == precision_cut {
                write!(f, "partition {partition}: {precision_cut} bits")?;
            } else {
                write!(
                    f,
                    "partition {partition}: {prev_precision_cut} up through {precision_cut} bits"
                )?;
            }
            if norm2_cut < f64::MAX {
                writeln!(f, " and norm2 {norm2_cut}")?;
            } else {
                writeln!(f)?;
            }
            prev_precision_cut = precision_cut + 1;
        }
        writeln!(
            f,
            "partition {}: {prev_precision_cut} bits and higher",
            self.p_cut.len()
        )?;
        Ok(())
    }
}
