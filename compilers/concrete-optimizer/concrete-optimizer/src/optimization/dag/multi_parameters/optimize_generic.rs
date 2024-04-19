use crate::dag::unparametrized::Dag;
use crate::optimization::config::{Config, SearchSpace};
use crate::optimization::dag::multi_parameters::keys_spec::CircuitSolution;
use crate::optimization::dag::multi_parameters::optimize::optimize_to_circuit_solution as native_optimize;
use crate::optimization::dag::solo_key::analyze;
use crate::optimization::dag::solo_key::optimize_generic::{max_precision, Encoding};
use crate::optimization::decomposition::PersistDecompCaches;
use crate::optimization::wop_atomic_pattern::optimize::optimize_to_circuit_solution as crt_optimize_no_dag;

use super::partition_cut::PartitionCut;

fn best_complexity_solution(native: CircuitSolution, crt: CircuitSolution) -> CircuitSolution {
    match (&native.is_feasible, &crt.is_feasible) {
        (true, true) => {
            // crt has 0 complexity in no lut case
            // so we always select native in this case
            if native.complexity <= crt.complexity || crt.complexity == 0.0 {
                native
            } else {
                crt
            }
        }
        (false, true) => crt,
        _ => native,
    }
}

fn crt_optimize(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    default_log_norm2_woppbs: f64,
    caches: &PersistDecompCaches,
) -> CircuitSolution {
    if analyze::has_round(dag) || analyze::has_unsafe_cast(dag) {
        return CircuitSolution::no_solution(
            "Crt does not support round/reinterpret_precision operator",
        );
    } // TODO: dag to params
    let max_precision = max_precision(dag);
    let nb_luts = analyze::lut_count_from_dag(dag);
    let worst_log_norm = analyze::worst_log_norm_for_wop(dag);
    let log_norm = default_log_norm2_woppbs.min(worst_log_norm);
    let nb_instr = dag.operators.len();
    crt_optimize_no_dag(
        max_precision as u64,
        nb_instr,
        nb_luts,
        config,
        log_norm,
        search_space,
        caches,
    )
}

pub fn optimize(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    encoding: Encoding,
    default_log_norm2_woppbs: f64,
    caches: &PersistDecompCaches,
    p_cut: &Option<PartitionCut>,
) -> CircuitSolution {
    let dag = dag.clone();
    let native = || native_optimize(&dag, config, search_space, caches, p_cut);
    let crt = || crt_optimize(&dag, config, search_space, default_log_norm2_woppbs, caches);
    match encoding {
        Encoding::Auto => best_complexity_solution(native(), crt()),
        Encoding::Native => native(),
        Encoding::Crt => crt(),
    }
}
