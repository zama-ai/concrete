use std::ops::RangeInclusive;

use concrete_commons::numeric::UnsignedInteger;

use crate::dag::operator::{Operator, Precision};
use crate::dag::unparametrized::OperationDag;
use crate::dag::unparametrized::UnparameterizedOperator;
use crate::optimization::atomic_pattern::Solution as WpSolution;
use crate::optimization::dag::solo_key::analyze;
use crate::optimization::dag::solo_key::optimize;
use crate::optimization::wop_atomic_pattern::optimize::optimize_one as wop_optimize;
use crate::optimization::wop_atomic_pattern::Solution as WopSolution;

const MINIMAL_WOP_PRECISION: Precision = 9;
const MAXIMAL_WOP_PRECISION: Precision = 16;
const WOP_PRECISIONS: RangeInclusive<Precision> = MINIMAL_WOP_PRECISION..=MAXIMAL_WOP_PRECISION;

pub enum Solution {
    WpSolution(WpSolution),
    WopSolution(WopSolution),
}

fn precision_op(op: &UnparameterizedOperator) -> Option<Precision> {
    match op {
        Operator::Input { out_precision, .. } | Operator::Lut { out_precision, .. } => {
            Some(*out_precision)
        }
        Operator::Dot { .. } | Operator::LevelledOp { .. } => None,
    }
}

fn max_precision(dag: &OperationDag) -> Precision {
    dag.operators
        .iter()
        .filter_map(precision_op)
        .max()
        .unwrap_or(0)
}

fn updated_global_p_error(nb_luts: u64, sol: WopSolution) -> WopSolution {
    let global_p_error = 1.0 - (1.0 - sol.p_error).powi(nb_luts as i32);
    WopSolution {
        global_p_error,
        ..sol
    }
}

pub fn optimize<W: UnsignedInteger>(
    dag: &OperationDag,
    security_level: u64,
    maximum_acceptable_error_probability: f64,
    glwe_log_polynomial_sizes: &[u64],
    glwe_dimensions: &[u64],
    internal_lwe_dimensions: &[u64],
    default_log_norm2_woppbs: f64,
) -> Option<Solution> {
    let max_precision = max_precision(dag);
    let nb_luts = analyze::lut_count_from_dag(dag);
    let has_luts = nb_luts != 0;
    if has_luts && WOP_PRECISIONS.contains(&max_precision) {
        let nb_luts = analyze::lut_count_from_dag(dag);
        let fallback_16b_precision = 16;
        let default_log_norm = default_log_norm2_woppbs;
        let worst_log_norm = analyze::worst_log_norm(dag);
        let log_norm = default_log_norm.min(worst_log_norm);
        let opt_sol = wop_optimize::<W>(
            fallback_16b_precision,
            security_level,
            log_norm,
            maximum_acceptable_error_probability,
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        )
        .best_solution;
        opt_sol.map(|sol| Solution::WopSolution(updated_global_p_error(nb_luts, sol)))
    } else {
        let opt_sol = optimize::optimize::<W>(
            dag,
            security_level,
            maximum_acceptable_error_probability,
            glwe_log_polynomial_sizes,
            glwe_dimensions,
            internal_lwe_dimensions,
        )
        .best_solution;
        opt_sol.map(Solution::WpSolution)
    }
}
