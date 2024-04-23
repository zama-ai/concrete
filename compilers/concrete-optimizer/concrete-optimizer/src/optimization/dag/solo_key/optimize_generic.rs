use crate::dag::operator::Precision;
use crate::dag::unparametrized::Dag;
use crate::noise_estimator::p_error::repeat_p_error;
use crate::optimization::atomic_pattern::Solution as WpSolution;
use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::dag::solo_key::{analyze, optimize};
use crate::optimization::decomposition::PersistDecompCaches;
use crate::optimization::wop_atomic_pattern::optimize::optimize_one as wop_optimize;
use crate::optimization::wop_atomic_pattern::Solution as WopSolution;

use super::analyze::{has_round, has_unsafe_cast};

pub enum Solution {
    WpSolution(WpSolution),
    WopSolution(WopSolution),
}

impl Solution {
    fn complexity(&self) -> f64 {
        match self {
            Self::WpSolution(v) => v.complexity,
            Self::WopSolution(v) => v.complexity,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Encoding {
    Auto,
    Native,
    Crt,
}

pub fn max_precision(dag: &Dag) -> Precision {
    dag.out_precisions.iter().copied().max().unwrap_or(0)
}

fn updated_global_p_error_and_complexity(nb_luts: u64, sol: WopSolution) -> WopSolution {
    let global_p_error = repeat_p_error(sol.p_error, nb_luts);
    let complexity = nb_luts as f64 * sol.complexity;
    WopSolution {
        complexity,
        global_p_error,
        ..sol
    }
}

fn best_complexity_solution(native: Option<Solution>, crt: Option<Solution>) -> Option<Solution> {
    match (&native, &crt) {
        (Some(s_native), Some(s_crt)) => {
            // crt has 0 complexity in no lut case
            // so we always select native in this case
            if s_native.complexity() <= s_crt.complexity() || s_crt.complexity() == 0.0 {
                native
            } else {
                crt
            }
        }
        (Some(_), None) => native,
        (None, Some(_)) => crt,
        (None, None) => None,
    }
}

fn optimize_with_wop_pbs(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    default_log_norm2_woppbs: f64,
    caches: &PersistDecompCaches,
) -> Option<WopSolution> {
    if has_round(dag) || has_unsafe_cast(dag) {
        return None;
    }
    let max_precision = max_precision(dag);
    let nb_luts = analyze::lut_count_from_dag(dag);
    let worst_log_norm = analyze::worst_log_norm_for_wop(dag);
    let log_norm = default_log_norm2_woppbs.min(worst_log_norm);
    wop_optimize(max_precision as u64, config, log_norm, search_space, caches)
        .best_solution
        .map(|sol| updated_global_p_error_and_complexity(nb_luts, sol))
}

pub fn optimize(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    encoding: Encoding,
    default_log_norm2_woppbs: f64,
    caches: &PersistDecompCaches,
) -> Option<Solution> {
    let native = || {
        optimize::optimize(dag, config, search_space, caches)
            .best_solution
            .map(Solution::WpSolution)
    };
    let crt = || {
        optimize_with_wop_pbs(dag, config, search_space, default_log_norm2_woppbs, caches)
            .map(Solution::WopSolution)
    };
    match encoding {
        Encoding::Auto => best_complexity_solution(native(), crt()),
        Encoding::Native => native(),
        Encoding::Crt => crt(),
    }
}
