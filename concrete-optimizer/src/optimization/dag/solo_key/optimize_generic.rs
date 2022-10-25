use crate::dag::operator::Precision;
use crate::dag::unparametrized::OperationDag;
use crate::noise_estimator::p_error::repeat_p_error;
use crate::optimization::atomic_pattern::Solution as WpSolution;
use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::dag::solo_key::{analyze, optimize};
use crate::optimization::decomposition::PersistDecompCaches;
use crate::optimization::wop_atomic_pattern::optimize::optimize_one as wop_optimize;
use crate::optimization::wop_atomic_pattern::Solution as WopSolution;

pub enum Solution {
    WpSolution(WpSolution),
    WopSolution(WopSolution),
}

fn max_precision(dag: &OperationDag) -> Precision {
    dag.out_precisions.iter().copied().max().unwrap_or(0)
}

fn updated_global_p_error(nb_luts: u64, sol: WopSolution) -> WopSolution {
    let global_p_error = repeat_p_error(sol.p_error, nb_luts);

    WopSolution {
        global_p_error,
        ..sol
    }
}

pub fn optimize(
    dag: &OperationDag,
    config: Config,
    search_space: &SearchSpace,
    default_log_norm2_woppbs: f64,
    caches: &PersistDecompCaches,
) -> Option<Solution> {
    let opt_sol = optimize::optimize(dag, config, search_space, caches).best_solution;
    if opt_sol.is_some() {
        return opt_sol.map(Solution::WpSolution);
    }
    let max_precision = max_precision(dag);
    let nb_luts = analyze::lut_count_from_dag(dag);
    let worst_log_norm = analyze::worst_log_norm(dag);
    let log_norm = default_log_norm2_woppbs.min(worst_log_norm);
    let opt_sol =
        wop_optimize(max_precision as u64, config, log_norm, search_space, caches).best_solution;
    opt_sol.map(|sol| Solution::WopSolution(updated_global_p_error(nb_luts, sol)))
}
