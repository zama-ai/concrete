// OPT: cache for fks and verified pareto
use concrete_cpu_noise_model::gaussian_noise::noise::modulus_switching::estimate_modulus_switching_noise_with_binary_key;

use crate::dag::unparametrized::Dag;
use crate::noise_estimator::error;
use crate::optimization;
use crate::optimization::config::{Config, NoiseBoundConfig, SearchSpace};
use crate::optimization::dag::multi_parameters::analyze::{analyze, AnalyzedDag};
use crate::optimization::dag::multi_parameters::fast_keyswitch;
use crate::optimization::dag::multi_parameters::fast_keyswitch::FksComplexityNoise;
use crate::optimization::dag::multi_parameters::operations_value::OperationsValue;
use crate::optimization::dag::solo_key::analyze::lut_count_from_dag;
use crate::optimization::dag::solo_key::optimize::optimize as optimize_mono;
use crate::optimization::decomposition::cmux::CmuxComplexityNoise;
use crate::optimization::decomposition::keyswitch::KsComplexityNoise;
use crate::optimization::decomposition::{cmux, keyswitch, DecompCaches, PersistDecompCaches};
use crate::parameters::GlweParameters;

use crate::optimization::dag::multi_parameters::complexity::Complexity;
use crate::optimization::dag::multi_parameters::feasible::Feasible;
use crate::optimization::dag::multi_parameters::partition_cut::PartitionCut;
use crate::optimization::dag::multi_parameters::partitions::PartitionIndex;
use crate::optimization::dag::multi_parameters::{analyze, keys_spec};
use crate::optimization::Err::{NoParametersFound, NotComposable};

use super::keys_spec::InstructionKeys;

const DEBUG: bool = false;

#[derive(Debug, Clone)]
pub struct MicroParameters {
    pub pbs: Vec<Option<CmuxComplexityNoise>>,
    pub ks: Vec<Vec<Option<KsComplexityNoise>>>,
    pub fks: Vec<Vec<Option<FksComplexityNoise>>>,
}

// Parameters optimized for 1 partition:
//  the partition pbs, all used ks for all partitions, a much fks as partition
struct PartialMicroParameters {
    pbs: CmuxComplexityNoise,
    ks: Vec<Vec<Option<KsComplexityNoise>>>,
    fks: Vec<Vec<Option<FksComplexityNoise>>>,
    p_error: f64,
    global_p_error: f64,
    complexity: f64,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct MacroParameters {
    pub glwe_params: GlweParameters,
    pub internal_dim: u64,
}

#[derive(Debug, Clone)]
pub struct Parameters {
    pub micro_params: MicroParameters,
    pub macro_params: Vec<Option<MacroParameters>>,
    is_lower_bound: bool,
    is_feasible: bool,
    pub p_error: f64,
    pub global_p_error: f64,
    pub complexity: f64,
}

#[derive(Debug, Clone)]
struct OperationsCV {
    variance: OperationsValue,
    cost: OperationsValue,
}

type KsSrc = PartitionIndex;
type KsDst = PartitionIndex;
type FksSrc = PartitionIndex;

#[inline(never)]
fn optimize_1_ks(
    ks_src: KsSrc,
    ks_dst: KsDst,
    ks_input_lwe_dim: u64,
    ks_pareto: &[KsComplexityNoise],
    operations: &mut OperationsCV,
    feasible: &Feasible,
    complexity: &Complexity,
    cut_complexity: f64,
) -> Option<KsComplexityNoise> {
    // find the first feasible (and less complex)
    let ks_max_variance = feasible.ks_max_feasible_variance(&operations.variance, ks_src, ks_dst);
    let ks_max_cost = complexity.ks_max_cost(cut_complexity, &operations.cost, ks_src, ks_dst);
    for &ks_quantity in ks_pareto {
        // variance is decreasing, complexity is increasing
        let ks_cost = ks_quantity.complexity(ks_input_lwe_dim);
        let ks_variance = ks_quantity.noise(ks_input_lwe_dim);
        if ks_cost > ks_max_cost {
            return None;
        }
        if ks_variance <= ks_max_variance {
            *operations.variance.ks(ks_src, ks_dst) = ks_variance;
            *operations.cost.ks(ks_src, ks_dst) = ks_cost;
            return Some(ks_quantity);
        }
    }
    None
}

fn optimize_many_independant_ks(
    macro_parameters: &[MacroParameters],
    ks_src: KsSrc,
    ks_input_lwe_dim: u64,
    ks_used: &[Vec<bool>],
    operations: &OperationsCV,
    feasible: &Feasible,
    complexity: &Complexity,
    caches: &mut keyswitch::Cache,
    cut_complexity: f64,
) -> Option<(Vec<(KsDst, KsComplexityNoise)>, OperationsCV)> {
    // all ks are independent since they appears in mutually exclusive variance constraints
    // only one ks can appear in a variance constraint,
    // we can obtain the best feasible by optimizing them separately since everything else is already chosen
    // at this point feasibility and minimal complexity has already been checked on lower bound
    // we know there a feasible solution and a better complexity solution
    // we just need to check if both properties at the same time occur
    debug_assert!(feasible.feasible(&operations.variance));
    debug_assert!(complexity.complexity(&operations.cost) <= cut_complexity);
    let mut operations = operations.clone();
    let mut ks_bests = Vec::with_capacity(macro_parameters.len());
    for (ks_dst, macro_dst) in macro_parameters.iter().enumerate() {
        let ks_dst = PartitionIndex(ks_dst);
        if !ks_used[ks_src.0][ks_dst.0] {
            continue;
        }
        let output_dim = macro_dst.internal_dim;
        let ks_pareto = caches.pareto_quantities(output_dim);
        let ks_best = optimize_1_ks(
            ks_src,
            ks_dst,
            ks_input_lwe_dim,
            ks_pareto,
            &mut operations,
            feasible,
            complexity,
            cut_complexity,
        )?; // abort if feasible but not with the right complexity
        ks_bests.push((ks_dst, ks_best));
    }
    Some((ks_bests, operations))
}

struct Best1FksAndManyKs {
    fks: Option<(FksSrc, FksComplexityNoise)>,
    many_ks: Vec<(KsDst, KsComplexityNoise)>,
}

#[allow(clippy::type_complexity)]
fn optimize_1_fks_and_all_compatible_ks(
    macro_parameters: &[MacroParameters],
    ks_used: &[Vec<bool>],
    fks_src: PartitionIndex,
    fks_dst: PartitionIndex,
    operations: &OperationsCV,
    feasible: &Feasible,
    complexity: &Complexity,
    caches: &mut keyswitch::Cache,
    cut_complexity: f64,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> Option<(Best1FksAndManyKs, OperationsCV)> {
    // At this point every thing else is known apart fks and ks
    let input_glwe = macro_parameters[fks_src.0].glwe_params;
    let output_glwe = macro_parameters[fks_dst.0].glwe_params;
    let output_lwe_dim = output_glwe.sample_extract_lwe_dimension();
    // OPT: have a separate cache for fks
    let ks_pareto = caches.pareto_quantities(output_lwe_dim).to_owned();
    // TODO: fast ks in the other direction as well
    let use_fast_ks = REAL_FAST_KS && input_glwe.sample_extract_lwe_dimension() >= output_lwe_dim;
    let ks_src = fks_dst;
    let ks_input_dim = macro_parameters[fks_dst.0]
        .glwe_params
        .sample_extract_lwe_dimension();
    let mut operations = operations.clone();
    let mut best_sol = None;
    let mut cut_complexity = cut_complexity;
    let same_dim = input_glwe == output_glwe;

    let fks_max_variance =
        feasible.fks_max_feasible_variance(&operations.variance, fks_src, fks_dst);
    let mut fks_max_cost =
        complexity.fks_max_cost(cut_complexity, &operations.cost, fks_src, fks_dst);
    for &ks_quantity in &ks_pareto {
        // OPT: add a pareto cache for fks
        let fks_quantity = if same_dim {
            FksComplexityNoise {
                decomp: ks_quantity.decomp,
                noise: 0.0,
                complexity: 0.0,
                src_glwe_param: input_glwe,
                dst_glwe_param: output_glwe,
            }
        } else if use_fast_ks {
            let noise = fast_keyswitch::noise(
                &ks_quantity,
                &input_glwe,
                &output_glwe,
                ciphertext_modulus_log,
                fft_precision,
            );
            let complexity =
                fast_keyswitch::complexity(&input_glwe, &output_glwe, ks_quantity.decomp.level);
            FksComplexityNoise {
                decomp: ks_quantity.decomp,
                noise,
                complexity,
                src_glwe_param: input_glwe,
                dst_glwe_param: output_glwe,
            }
        } else {
            let noise = ks_quantity.noise(input_glwe.sample_extract_lwe_dimension());
            let complexity = ks_quantity.complexity(input_glwe.sample_extract_lwe_dimension());
            FksComplexityNoise {
                decomp: ks_quantity.decomp,
                noise,
                complexity,
                src_glwe_param: input_glwe,
                dst_glwe_param: output_glwe,
            }
        };

        if fks_quantity.complexity > fks_max_cost {
            // complexity is strictly increasing by level
            // next complexity will be worse
            return best_sol;
        }

        if fks_quantity.noise > fks_max_variance {
            continue;
        }

        *operations.cost.fks(fks_src, fks_dst) = fks_quantity.complexity;
        *operations.variance.fks(fks_src, fks_dst) = fks_quantity.noise;

        let sol = optimize_many_independant_ks(
            macro_parameters,
            ks_src,
            ks_input_dim,
            ks_used,
            &operations,
            feasible,
            complexity,
            caches,
            cut_complexity,
        );
        if sol.is_none() {
            continue;
        }
        let (best_many_ks, operations) = sol.unwrap();
        let cost = complexity.complexity(&operations.cost);
        if cost > cut_complexity {
            continue;
        }
        cut_complexity = cost;
        fks_max_cost = complexity.fks_max_cost(cut_complexity, &operations.cost, fks_src, fks_dst);
        // COULD: handle complexity tie
        let bests = Best1FksAndManyKs {
            fks: Some((fks_src, fks_quantity)),
            many_ks: best_many_ks,
        };
        best_sol = Some((bests, operations));
        if same_dim {
            break;
        }
    }
    best_sol
}

fn optimize_dst_exclusive_fks_subset_and_all_ks(
    macro_parameters: &[MacroParameters],
    fks_paretos: &[Option<FksSrc>],
    ks_used: &[Vec<bool>],
    operations: &OperationsCV,
    feasible: &Feasible,
    complexity: &Complexity,
    caches: &mut keyswitch::Cache,
    cut_complexity: f64,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> Option<(Vec<Best1FksAndManyKs>, OperationsCV)> {
    // All fks subgroup can be optimized independently
    let mut acc_operations = operations.clone();
    let mut result = vec![];
    result.reserve_exact(fks_paretos.len());
    for (fks_dst, maybe_fks_pareto) in fks_paretos.iter().enumerate() {
        let ks_src = fks_dst;
        let ks_input_lwe_dim = macro_parameters[fks_dst]
            .glwe_params
            .sample_extract_lwe_dimension();
        if let Some(fks_src) = maybe_fks_pareto {
            let (bests, operations) = optimize_1_fks_and_all_compatible_ks(
                macro_parameters,
                ks_used,
                *fks_src,
                PartitionIndex(fks_dst),
                &acc_operations,
                feasible,
                complexity,
                caches,
                cut_complexity,
                ciphertext_modulus_log,
                fft_precision,
            )?;
            result.push(bests);
            _ = std::mem::replace(&mut acc_operations, operations);
        } else {
            // There is no fks to optimize
            let (many_ks, operations) = optimize_many_independant_ks(
                macro_parameters,
                PartitionIndex(ks_src),
                ks_input_lwe_dim,
                ks_used,
                &acc_operations,
                feasible,
                complexity,
                caches,
                cut_complexity,
            )?;
            result.push(Best1FksAndManyKs { fks: None, many_ks });
            _ = std::mem::replace(&mut acc_operations, operations);
        }
    }
    Some((result, acc_operations))
}

fn optimize_1_cmux_and_dst_exclusive_fks_subset_and_all_ks(
    partition: PartitionIndex,
    macro_parameters: &[MacroParameters],
    internal_dim: u64,
    cmux_pareto: &[CmuxComplexityNoise],
    fks_paretos: &[Option<FksSrc>],
    ks_used: &[Vec<bool>],
    operations: &OperationsCV,
    feasible: &Feasible,
    complexity: &Complexity,
    caches: &mut keyswitch::Cache,
    cut_complexity: f64,
    best_p_error: f64,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) -> Option<PartialMicroParameters> {
    let mut operations = operations.clone();
    let mut best_sol = None;
    let mut best_sol_complexity = cut_complexity;
    let mut best_sol_p_error = best_p_error;
    let mut best_sol_global_p_error = 1.0;

    let pbs_max_feasible_variance =
        feasible.pbs_max_feasible_variance(&operations.variance, partition);
    for &cmux_quantity in cmux_pareto {
        // increasing complexity, decreasing variance

        // Lower bounds cuts
        let pbs_cost = cmux_quantity.complexity_br(internal_dim);
        *operations.cost.pbs(partition) = pbs_cost;
        let lower_cost = complexity.complexity(&operations.cost);
        if lower_cost > best_sol_complexity {
            continue;
        }

        let pbs_variance = cmux_quantity.noise_br(internal_dim);
        if pbs_variance > pbs_max_feasible_variance {
            continue;
        }

        *operations.variance.pbs(partition) = pbs_variance;
        let sol = optimize_dst_exclusive_fks_subset_and_all_ks(
            macro_parameters,
            fks_paretos,
            ks_used,
            &operations,
            feasible,
            complexity,
            caches,
            best_sol_complexity,
            ciphertext_modulus_log,
            fft_precision,
        );
        if sol.is_none() {
            continue;
        }

        let (best_fks_ks, operations) = sol.unwrap();
        let cost = complexity.complexity(&operations.cost);
        if cost > best_sol_complexity {
            continue;
        };
        let p_error = feasible.p_error(&operations.variance);
        #[allow(clippy::float_cmp)]
        if cost == best_sol_complexity && p_error >= best_sol_p_error {
            continue;
        }
        best_sol_complexity = cost;
        best_sol_p_error = p_error;
        best_sol_global_p_error = feasible.global_p_error(&operations.variance);
        best_sol = Some((cmux_quantity, best_fks_ks));
    }
    if best_sol.is_none() {
        return None;
    }
    let nb_partitions = macro_parameters.len();
    let (cmux_quantity, best_fks_ks) = best_sol.unwrap();
    let mut fks = vec![vec![None; nb_partitions]; nb_partitions];
    let mut ks = vec![vec![None; nb_partitions]; nb_partitions];
    for (fks_dst, one_best_fks_ks) in best_fks_ks.iter().enumerate() {
        if let Some((fks_src, sol_fks)) = one_best_fks_ks.fks {
            fks[fks_src.0][fks_dst] = Some(sol_fks);
        }
        for (ks_dst, sol_ks) in &one_best_fks_ks.many_ks {
            ks[fks_dst][ks_dst.0] = Some(*sol_ks);
        }
    }
    Some(PartialMicroParameters {
        pbs: cmux_quantity,
        fks,
        ks,
        p_error: best_sol_p_error,
        global_p_error: best_sol_global_p_error,
        complexity: best_sol_complexity,
    })
}

fn apply_all_ks_lower_bound(
    caches: &mut keyswitch::Cache,
    nb_partitions: usize,
    macro_parameters: &[MacroParameters],
    used_tlu_keyswitch: &[Vec<bool>],
    operations: &mut OperationsCV,
) {
    for (src, dst) in cross_partition(nb_partitions) {
        if !used_tlu_keyswitch[src.0][dst.0] {
            continue;
        }
        let in_glwe_params = macro_parameters[src.0].glwe_params;
        let out_internal_dim = macro_parameters[dst.0].internal_dim;
        let ks_pareto = caches.pareto_quantities(out_internal_dim);
        let in_lwe_dim = in_glwe_params.sample_extract_lwe_dimension();
        *operations.variance.ks(src, dst) = keyswitch::lowest_noise_ks(ks_pareto, in_lwe_dim);
        *operations.cost.ks(src, dst) = keyswitch::lowest_complexity_ks(ks_pareto, in_lwe_dim);
    }
}

fn apply_fks_variance_and_cost_or_lower_bound(
    caches: &mut keyswitch::Cache,
    nb_partitions: usize,
    macro_parameters: &[MacroParameters],
    initial_fks: &[Vec<Option<FksComplexityNoise>>],
    fks_to_optimize: &[Option<FksSrc>],
    used_conversion_keyswitch: &[Vec<bool>],
    operations: &mut OperationsCV,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
) {
    for (src, dst) in cross_partition(nb_partitions) {
        if !used_conversion_keyswitch[src.0][dst.0] {
            continue;
        }
        let input_glwe = &macro_parameters[src.0].glwe_params;
        let output_glwe = &macro_parameters[dst.0].glwe_params;
        if input_glwe == output_glwe {
            *operations.variance.fks(src, dst) = 0.0;
            *operations.cost.fks(src, dst) = 0.0;
            continue;
        }
        // if an optimized fks is applicable and is not to be optimized
        // we use the already optimized fks instead of a lower bound
        if let Some(fks) = initial_fks[src.0][dst.0] {
            let to_be_optimized = fks_to_optimize[src.0].map_or(false, |fdst| dst == fdst);
            if !to_be_optimized {
                if input_glwe == &fks.src_glwe_param && output_glwe == &fks.dst_glwe_param {
                    *operations.variance.fks(src, dst) = fks.noise;
                    *operations.cost.fks(src, dst) = fks.complexity;
                }
                continue;
            }
        }
        let ks_pareto = caches.pareto_quantities(output_glwe.sample_extract_lwe_dimension());
        let use_fast_ks = REAL_FAST_KS
            && input_glwe.sample_extract_lwe_dimension()
                >= output_glwe.sample_extract_lwe_dimension();
        let cost = if use_fast_ks {
            fast_keyswitch::complexity(input_glwe, output_glwe, ks_pareto[0].decomp.level)
        } else {
            keyswitch::lowest_complexity_ks(ks_pareto, input_glwe.sample_extract_lwe_dimension())
        };
        *operations.cost.fks(src, dst) = cost;
        let mut variance_min = f64::INFINITY;
        // TODO: use a pareto front to avoid that loop
        if use_fast_ks {
            for ks_q in ks_pareto {
                let variance = fast_keyswitch::noise(
                    ks_q,
                    input_glwe,
                    output_glwe,
                    ciphertext_modulus_log,
                    fft_precision,
                );
                variance_min = variance_min.min(variance);
            }
        } else {
            variance_min =
                keyswitch::lowest_noise_ks(ks_pareto, input_glwe.sample_extract_lwe_dimension());
        }
        *operations.variance.fks(src, dst) = variance_min;
    }
}

fn apply_partitions_input_and_modulus_variance_and_cost(
    ciphertext_modulus_log: u32,
    security_level: u64,
    nb_partitions: usize,
    macro_parameters: &[MacroParameters],
    partition: PartitionIndex,
    input_variance: f64,
    variance_modulus_switching: f64,
    operations: &mut OperationsCV,
) {
    for i in PartitionIndex::range(0, nb_partitions) {
        let (input_variance, variance_modulus_switching) =
            if macro_parameters[i.0] == macro_parameters[partition.0] {
                (input_variance, variance_modulus_switching)
            } else {
                let input_variance = macro_parameters[i.0]
                    .glwe_params
                    .minimal_variance(ciphertext_modulus_log, security_level);
                let variance_modulus_switching = estimate_modulus_switching_noise_with_binary_key(
                    macro_parameters[i.0].internal_dim,
                    macro_parameters[i.0].glwe_params.log2_polynomial_size,
                    ciphertext_modulus_log,
                );
                (input_variance, variance_modulus_switching)
            };
        *operations.variance.input(i) = input_variance;
        *operations.variance.modulus_switching(i) = variance_modulus_switching;
    }
}

fn apply_pbs_variance_and_cost_or_lower_bounds(
    caches: &mut cmux::Cache,
    macro_parameters: &[MacroParameters],
    initial_pbs: &[Option<CmuxComplexityNoise>],
    partition: PartitionIndex,
    operations: &mut OperationsCV,
) {
    // setting already chosen pbs and lower bounds
    for (i, pbs) in initial_pbs.iter().enumerate() {
        let i = PartitionIndex(i);
        let pbs = if i == partition { &None } else { pbs };
        if let Some(pbs) = pbs {
            let internal_dim = macro_parameters[i.0].internal_dim;
            *operations.variance.pbs(i) = pbs.noise_br(internal_dim);
            *operations.cost.pbs(i) = pbs.complexity_br(internal_dim);
        } else {
            // OPT: Most values could be shared on first optimize_macro
            let in_internal_dim = macro_parameters[i.0].internal_dim;
            let out_glwe_params = macro_parameters[i.0].glwe_params;
            let variance_min =
                cmux::lowest_noise_br(caches.pareto_quantities(out_glwe_params), in_internal_dim);
            *operations.variance.pbs(i) = variance_min;
            *operations.cost.pbs(i) = 0.0;
        }
    }
}

fn fks_to_optimize(
    nb_partitions: usize,
    used_conversion_keyswitch: &[Vec<bool>],
    optimized_partition: PartitionIndex,
) -> Vec<Option<FksSrc>> {
    // Prepare a subset fks pareto to optimize: real, lower, bound or unused (fake)
    // We only take 1 fks pareto fks[_->dst] with different dst partition for each dst, since they can be optimized independently.
    // I.e. They appears only in constraints with ks[fks_dst->_].
    // When fks is unused a None is used to keep the same loop structure.
    let mut fks_paretos: Vec<Option<_>> = vec![];
    fks_paretos.reserve_exact(nb_partitions);
    for fks_dst in PartitionIndex::range(0, nb_partitions) {
        // find the i-th valid fks_src
        let fks_src = if used_conversion_keyswitch[optimized_partition.0][fks_dst.0] {
            Some(optimized_partition)
        } else {
            let mut count_used: usize = 0;
            let mut fks_src = None;
            #[allow(clippy::needless_range_loop)]
            for src in PartitionIndex::range(0, nb_partitions) {
                let used = used_conversion_keyswitch[src.0][fks_dst.0];
                if used && count_used == optimized_partition.0 {
                    fks_src = Some(src);
                    break;
                }
                count_used += used as usize;
            }
            if fks_src.is_none() && count_used > 0 {
                let n_th = optimized_partition.0 % count_used;
                count_used = 0;
                #[allow(clippy::needless_range_loop)]
                for src in PartitionIndex::range(0, nb_partitions) {
                    let used = used_conversion_keyswitch[src.0][fks_dst.0];
                    if used && count_used == n_th {
                        fks_src = Some(src);
                        break;
                    }
                }
            }
            fks_src
        };
        fks_paretos.push(fks_src);
    }
    fks_paretos
}

// In case fast ks are not used
pub const REAL_FAST_KS: bool = false;

#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_lines)]
fn optimize_macro(
    security_level: u64,
    ciphertext_modulus_log: u32,
    fft_precision: u32,
    search_space: &SearchSpace,
    partition: PartitionIndex,
    used_tlu_keyswitch: &[Vec<bool>],
    used_conversion_keyswitch: &[Vec<bool>],
    feasible: &Feasible,
    complexity: &Complexity,
    caches: &mut DecompCaches,
    init_parameters: &Parameters,
    best_complexity: f64,
    best_p_error: f64,
) -> Parameters {
    let nb_partitions = init_parameters.macro_params.len();
    assert!(partition.0 < nb_partitions);

    let variance_modulus_switching_of = |glwe_log2_poly_size, internal_lwe_dimensions| {
        estimate_modulus_switching_noise_with_binary_key(
            internal_lwe_dimensions,
            glwe_log2_poly_size,
            ciphertext_modulus_log,
        )
    };

    let mut best_parameters = init_parameters.clone();
    let mut best_complexity = best_complexity;
    let mut best_p_error = best_p_error;
    let mut best_partition_p_error = f64::INFINITY;

    let fks_to_optimize = fks_to_optimize(nb_partitions, used_conversion_keyswitch, partition);
    let operations = OperationsCV {
        variance: feasible.zero_variance(),
        cost: complexity.zero_cost(),
    };
    let partition_feasible = feasible.filter_constraints(partition);

    let glwe_params_domain = search_space.glwe_dimensions.iter().flat_map(|a| {
        search_space
            .glwe_log_polynomial_sizes
            .iter()
            .map(|b| (*a, *b))
    });
    let mut lb_message = None;
    for (glwe_dimension, log2_polynomial_size) in glwe_params_domain {
        let glwe_params = GlweParameters {
            log2_polynomial_size,
            glwe_dimension,
        };

        let input_variance = glwe_params.minimal_variance(ciphertext_modulus_log, security_level);
        if glwe_dimension == 1 && log2_polynomial_size == 8 {
            // this is insecure and so minimal variance will be above 1
            assert!(input_variance > 1.0);
            continue;
        }

        for &internal_dim in &search_space.internal_lwe_dimensions {
            let mut operations = operations.clone();
            // OPT: fast linear noise_modulus_switching
            let variance_modulus_switching =
                variance_modulus_switching_of(log2_polynomial_size, internal_dim);

            let macro_param_partition = MacroParameters {
                glwe_params,
                internal_dim,
            };

            // Heuristic to fill missing macro parameters
            let macros: Vec<_> = PartitionIndex::range(0, nb_partitions)
                .map(|i| {
                    if i == partition {
                        macro_param_partition
                    } else {
                        init_parameters.macro_params[i.0].unwrap_or(macro_param_partition)
                    }
                })
                .collect();

            // OPT: could be done once and than partially updated
            apply_partitions_input_and_modulus_variance_and_cost(
                ciphertext_modulus_log,
                security_level,
                nb_partitions,
                &macros,
                partition,
                input_variance,
                variance_modulus_switching,
                &mut operations,
            );

            if best_parameters.is_feasible && !feasible.feasible(&operations.variance) {
                // noise_modulus_switching is increasing with internal_dim so we can cut
                // but as long as nothing feasible as been found we don't break to improve feasibility
                break;
            }

            if complexity.complexity(&operations.cost) > best_complexity {
                continue;
            }

            // setting already chosen pbs and lower bounds
            // OPT: could be done once and than partially updated
            apply_pbs_variance_and_cost_or_lower_bounds(
                &mut caches.cmux,
                &macros,
                &init_parameters.micro_params.pbs,
                partition,
                &mut operations,
            );

            // OPT: could be done once and than partially updated
            apply_all_ks_lower_bound(
                &mut caches.keyswitch,
                nb_partitions,
                &macros,
                used_tlu_keyswitch,
                &mut operations,
            );
            // OPT: could be done once and than partially updated
            apply_fks_variance_and_cost_or_lower_bound(
                &mut caches.keyswitch,
                nb_partitions,
                &macros,
                &init_parameters.micro_params.fks,
                &fks_to_optimize,
                used_conversion_keyswitch,
                &mut operations,
                ciphertext_modulus_log,
                fft_precision,
            );

            let non_feasible = !feasible.feasible(&operations.variance);
            if best_parameters.is_feasible && non_feasible {
                continue;
            }

            if complexity.complexity(&operations.cost) > best_complexity {
                continue;
            }

            let cmux_pareto = caches.cmux.pareto_quantities(glwe_params);

            if non_feasible {
                lb_message = Some("Non feasible");
                // here we optimize for feasibility only
                // if nothing is feasible, it will give improves feasability for later iterations
                let mut macro_params = init_parameters.macro_params.clone();
                macro_params[partition.0] = Some(MacroParameters {
                    glwe_params,
                    internal_dim,
                });
                // optimize the feasibility only, takes all lower bounds on variance
                // this selects both macro parameters and pbs (lowest variance) for this partition
                let complexity = f64::INFINITY;
                let cmux_params = cmux::lowest_noise(cmux_pareto);
                let partition_p_error = partition_feasible.p_error(&operations.variance);
                if partition_p_error >= best_partition_p_error {
                    continue;
                }
                best_partition_p_error = partition_p_error;
                let p_error = feasible.p_error(&operations.variance);
                let global_p_error = feasible.global_p_error(&operations.variance);
                let mut pbs = init_parameters.micro_params.pbs.clone();
                pbs[partition.0] = Some(cmux_params);
                let micro_params = MicroParameters {
                    pbs,
                    ks: vec![vec![None; nb_partitions]; nb_partitions],
                    fks: vec![vec![None; nb_partitions]; nb_partitions],
                };
                best_parameters = Parameters {
                    p_error,
                    global_p_error,
                    complexity,
                    micro_params,
                    macro_params,
                    is_lower_bound: true,
                    is_feasible: false,
                };
                continue;
            }

            if complexity.complexity(&operations.cost) > best_complexity {
                continue;
            }

            let micro_opt = optimize_1_cmux_and_dst_exclusive_fks_subset_and_all_ks(
                partition,
                &macros,
                internal_dim,
                cmux_pareto,
                &fks_to_optimize,
                used_tlu_keyswitch,
                &operations,
                feasible,
                complexity,
                &mut caches.keyswitch,
                best_complexity,
                best_p_error,
                ciphertext_modulus_log,
                fft_precision,
            );
            if let Some(some_micro_params) = micro_opt {
                // erase macros and all fks that can't be real
                // set global is_lower_bound here, if any parameter is missing this is lower bound
                // optimize_micro has already checked for best-ness
                lb_message = None;
                let mut macro_params = init_parameters.macro_params.clone();
                macro_params[partition.0] = Some(macro_param_partition);
                let mut is_lower_bound = macro_params.iter().any(Option::is_none);
                if is_lower_bound {
                    lb_message = Some("is_lower_bound due to missing macro parameter");
                }
                // copy back pbs from other partition
                let mut all_pbs = init_parameters.micro_params.pbs.clone();
                all_pbs[partition.0] = Some(some_micro_params.pbs);
                let mut all_fks = init_parameters.micro_params.fks.clone();
                for (dst_partition, maybe_fks) in fks_to_optimize.iter().enumerate() {
                    let dst_partition = PartitionIndex(dst_partition);
                    if let &Some(src_partition) = maybe_fks {
                        all_fks[src_partition.0][dst_partition.0] =
                            some_micro_params.fks[src_partition.0][dst_partition.0];
                        assert!(used_conversion_keyswitch[src_partition.0][dst_partition.0]);
                        assert!(all_fks[src_partition.0][dst_partition.0].is_some());
                    }
                }
                // As all fks cannot be re-optimized in some case, we need to check previous ones are still valid.
                for (src_partition, dst_partition) in cross_partition(nb_partitions) {
                    if !used_conversion_keyswitch[src_partition.0][dst_partition.0] {
                        continue;
                    }
                    let fks = &all_fks[src_partition.0][dst_partition.0];
                    if !is_lower_bound && fks.is_none() {
                        lb_message = Some("is_lower_bound due to missing fast keyswitch parameter");
                        is_lower_bound = true;
                    }
                    let src_glwe_param = macro_params[src_partition.0].map(|p| p.glwe_params);
                    let dst_glwe_param = macro_params[dst_partition.0].map(|p| p.glwe_params);
                    let src_glwe_param_stable = src_glwe_param == fks.map(|p| p.src_glwe_param);
                    let dst_glwe_param_stable = dst_glwe_param == fks.map(|p| p.dst_glwe_param);
                    if src_glwe_param_stable && dst_glwe_param_stable {
                        continue;
                    }
                    if !is_lower_bound {
                        lb_message = Some("is_lower_bound due to changing others fks macro param");
                    }
                    all_fks[src_partition.0][dst_partition.0] = None;
                    is_lower_bound = true;
                }
                let micro_params = MicroParameters {
                    pbs: all_pbs,
                    ks: some_micro_params.ks,
                    fks: all_fks,
                };
                best_complexity = some_micro_params.complexity;
                best_p_error = some_micro_params.p_error;
                best_parameters = Parameters {
                    p_error: best_p_error,
                    global_p_error: some_micro_params.global_p_error,
                    complexity: best_complexity,
                    micro_params,
                    macro_params,
                    is_lower_bound,
                    is_feasible: true,
                };
            } else {
                // the macro parameters are feasible
                // but the complexity is not good enough due to previous feasible solution
                assert!(best_parameters.is_feasible);
            }
        }
    }
    if DEBUG && lb_message.is_some() {
        eprintln!("{}", lb_message.unwrap());
    }
    best_parameters
}

fn cross_partition(nb_partitions: usize) -> impl Iterator<Item = (PartitionIndex, PartitionIndex)> {
    PartitionIndex::range(0, nb_partitions)
        .flat_map(move |a| PartitionIndex::range(0, nb_partitions).map(move |b| (a, b)))
}

#[allow(clippy::too_many_lines, clippy::missing_errors_doc)]
pub fn optimize(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    persistent_caches: &PersistDecompCaches,
    p_cut: &Option<PartitionCut>,
    default_partition: PartitionIndex,
) -> optimization::Result<(AnalyzedDag, Parameters)> {
    let ciphertext_modulus_log = config.ciphertext_modulus_log;
    let fft_precision = config.fft_precision;
    let security_level = config.security_level;
    let noise_config = NoiseBoundConfig {
        security_level,
        maximum_acceptable_error_probability: config.maximum_acceptable_error_probability,
        ciphertext_modulus_log,
    };

    let dag = analyze(dag, &noise_config, p_cut, default_partition)?;
    let kappa =
        error::sigma_scale_of_error_probability(config.maximum_acceptable_error_probability);

    let mut caches = persistent_caches.caches();

    let feasible = Feasible::of(&dag.variance_constraints, kappa, None).compressed();
    let complexity = Complexity::of(&dag.operations_count).compressed();
    let used_tlu_keyswitch = used_tlu_keyswitch(&dag);
    let used_conversion_keyswitch = used_conversion_keyswitch(&dag);

    let nb_partitions = dag.nb_partitions;
    let init_parameters = Parameters {
        is_lower_bound: false,
        is_feasible: false,
        macro_params: vec![None; nb_partitions],
        micro_params: MicroParameters {
            pbs: vec![None; nb_partitions],
            ks: vec![vec![None; nb_partitions]; nb_partitions],
            fks: vec![vec![None; nb_partitions]; nb_partitions],
        },
        p_error: 1.0,
        global_p_error: 1.0,
        complexity: f64::INFINITY,
    };

    let mut params = init_parameters;
    let mut best_complexity = f64::INFINITY;
    let mut best_p_error = f64::INFINITY;

    let mut fix_point = params.clone();
    let mut best_params: Option<Parameters> = None;
    for iter in 0..=10 {
        for partition in PartitionIndex::range(0, nb_partitions).rev() {
            let new_params = optimize_macro(
                security_level,
                ciphertext_modulus_log,
                fft_precision,
                search_space,
                partition,
                &used_tlu_keyswitch,
                &used_conversion_keyswitch,
                &feasible,
                &complexity,
                &mut caches,
                &params,
                best_complexity,
                best_p_error,
            );
            assert!(
                new_params.is_feasible || !params.is_feasible,
                "Cannot degrade feasibility"
            );
            params = new_params;
            if !params.is_feasible {
                if nb_partitions == 1 {
                    return Err(NoParametersFound);
                }
                if DEBUG {
                    eprintln!(
                        "Intermediate non feasible solution {iter} : {partition} : {}",
                        params.p_error
                    );
                }
                continue;
            }
            if params.is_lower_bound {
                if DEBUG {
                    eprintln!(
                        "Lower bound solution it:{iter} : part:{partition} : {} {} lb:{}",
                        params.p_error, params.complexity, params.is_lower_bound
                    );
                }
                continue;
            }
            if DEBUG {
                eprintln!(
                    "Feasible solution {iter} : {partition} : {} {} {}",
                    params.p_error, params.complexity, params.is_lower_bound
                );
            }
            if params.complexity < best_complexity {
                best_complexity = params.complexity;
                best_p_error = params.p_error;
                best_params = Some(params.clone());
            }
        }
        if nb_partitions == 1 {
            break;
        }
        // OPT: could be detected sooner
        #[allow(clippy::float_cmp)]
        if fix_point.complexity == params.complexity
            && fix_point.p_error == params.p_error
            && fix_point.macro_params == params.macro_params
        {
            if DEBUG {
                eprintln!("Fix point reached at {iter}");
            }
            break;
        }
        fix_point = params.clone();
    }
    if best_params.is_none() {
        return Err(NoParametersFound);
    }
    let best_params = best_params.unwrap();
    sanity_check(
        &best_params,
        &used_conversion_keyswitch,
        &used_tlu_keyswitch,
        ciphertext_modulus_log,
        security_level,
        &feasible,
        &complexity,
    );
    Ok((dag, best_params))
}

fn used_tlu_keyswitch(dag: &AnalyzedDag) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; dag.nb_partitions]; dag.nb_partitions];
    for (src_partition, dst_partition) in cross_partition(dag.nb_partitions) {
        for constraint in &dag.variance_constraints {
            if constraint
                .variance
                .coeff_keyswitch_to_small(src_partition, dst_partition)
                != 0.0
            {
                result[src_partition.0][dst_partition.0] = true;
                break;
            }
        }
    }
    result
}

fn used_conversion_keyswitch(dag: &AnalyzedDag) -> Vec<Vec<bool>> {
    let mut result = vec![vec![false; dag.nb_partitions]; dag.nb_partitions];
    for (src_partition, dst_partition) in cross_partition(dag.nb_partitions) {
        for constraint in &dag.variance_constraints {
            if constraint
                .variance
                .coeff_partition_keyswitch_to_big(src_partition, dst_partition)
                != 0.0
            {
                result[src_partition.0][dst_partition.0] = true;
                break;
            }
        }
    }
    result
}

#[allow(clippy::float_cmp)]
fn sanity_check(
    params: &Parameters,
    used_conversion_keyswitch: &[Vec<bool>],
    used_tlu_keyswitch: &[Vec<bool>],
    ciphertext_modulus_log: u32,
    security_level: u64,
    feasible: &Feasible,
    complexity: &Complexity,
) {
    assert!(params.is_feasible);
    assert!(
        !params.is_lower_bound,
        "Sanity check:lower_bound: cannot return a partial solution"
    );
    let nb_partitions = params.macro_params.len();
    let mut operations = OperationsCV {
        variance: feasible.zero_variance(),
        cost: complexity.zero_cost(),
    };
    let micro_params = &params.micro_params;
    for partition in PartitionIndex::range(0, nb_partitions) {
        let partition_macro = params.macro_params[partition.0].unwrap();
        let glwe_param = partition_macro.glwe_params;
        let internal_dim = partition_macro.internal_dim;
        let input_variance = glwe_param.minimal_variance(ciphertext_modulus_log, security_level);
        let variance_modulus_switching = estimate_modulus_switching_noise_with_binary_key(
            internal_dim,
            glwe_param.log2_polynomial_size,
            ciphertext_modulus_log,
        );
        *operations.variance.input(partition) = input_variance;
        *operations.variance.modulus_switching(partition) = variance_modulus_switching;
        if let Some(pbs) = micro_params.pbs[partition.0] {
            *operations.variance.pbs(partition) = pbs.noise_br(internal_dim);
            *operations.cost.pbs(partition) = pbs.complexity_br(internal_dim);
        } else {
            *operations.variance.pbs(partition) = f64::MAX;
            *operations.cost.pbs(partition) = f64::MAX;
        }
        for src_partition in PartitionIndex::range(0, nb_partitions) {
            let src_partition_macro = params.macro_params[src_partition.0].unwrap();
            let src_glwe_param = src_partition_macro.glwe_params;
            let src_lwe_dim = src_glwe_param.sample_extract_lwe_dimension();
            if let Some(ks) = micro_params.ks[src_partition.0][partition.0] {
                assert!(
                    used_tlu_keyswitch[src_partition.0][partition.0],
                    "Superflous ks[{src_partition}->{partition}]"
                );
                *operations.variance.ks(src_partition, partition) = ks.noise(src_lwe_dim);
                *operations.cost.ks(src_partition, partition) = ks.complexity(src_lwe_dim);
            } else {
                assert!(
                    !used_tlu_keyswitch[src_partition.0][partition.0],
                    "Missing ks[{src_partition}->{partition}]"
                );
                *operations.variance.ks(src_partition, partition) = f64::MAX;
                *operations.cost.ks(src_partition, partition) = f64::MAX;
            }
            if let Some(fks) = micro_params.fks[src_partition.0][partition.0] {
                assert!(
                    used_conversion_keyswitch[src_partition.0][partition.0],
                    "Superflous fks[{src_partition}->{partition}]"
                );
                *operations.variance.fks(src_partition, partition) = fks.noise;
                *operations.cost.fks(src_partition, partition) = fks.complexity;
            } else {
                assert!(
                    !used_conversion_keyswitch[src_partition.0][partition.0],
                    "Missing fks[{src_partition}->{partition}]"
                );
                *operations.variance.fks(src_partition, partition) = f64::MAX;
                *operations.cost.fks(src_partition, partition) = f64::MAX;
            }
        }
    }
    #[allow(clippy::float_cmp)]
    {
        assert!(feasible.feasible(&operations.variance));
        assert!(params.p_error == feasible.p_error(&operations.variance));
        assert!(params.complexity == complexity.complexity(&operations.cost));
        assert!(params.global_p_error == feasible.global_p_error(&operations.variance));
    }
}

pub fn optimize_to_circuit_solution(
    dag: &Dag,
    config: Config,
    search_space: &SearchSpace,
    persistent_caches: &PersistDecompCaches,
    p_cut: &Option<PartitionCut>,
) -> keys_spec::CircuitSolution {
    if lut_count_from_dag(dag) == 0 {
        // If there are no lut in the dag the noise is never refresh so the dag cannot be composable
        if dag.is_composed() {
            return keys_spec::CircuitSolution::no_solution(
                NotComposable("No luts in the circuit.".into()).to_string(),
            );
        }
        let nb_instr = dag.operators.len();
        if let Some(sol) = optimize_mono(dag, config, search_space, persistent_caches).best_solution
        {
            return keys_spec::CircuitSolution::from_native_solution(sol, nb_instr);
        }
        return keys_spec::CircuitSolution::no_solution(NoParametersFound.to_string());
    }
    let default_partition = PartitionIndex::FIRST;
    let dag_and_params = optimize(
        dag,
        config,
        search_space,
        persistent_caches,
        p_cut,
        default_partition,
    );
    #[allow(clippy::option_if_let_else)]
    match dag_and_params {
        Err(e) => keys_spec::CircuitSolution::no_solution(e.to_string()),
        Ok((dag, params)) => {
            let ext_keys = keys_spec::ExpandedCircuitKeys::of(&params);
            let instructions_keys = analyze::original_instrs_partition(&dag, &ext_keys);
            let (ext_keys, instructions_keys) = if config.key_sharing {
                let (ext_keys, key_sharing) = ext_keys.shared_keys();
                let instructions_keys =
                    InstructionKeys::shared_keys(&instructions_keys, &key_sharing);
                (ext_keys, instructions_keys)
            } else {
                (ext_keys, instructions_keys)
            };
            let circuit_keys = ext_keys.compacted();
            keys_spec::CircuitSolution {
                circuit_keys,
                instructions_keys,
                crt_decomposition: vec![],
                complexity: params.complexity,
                p_error: params.p_error,
                global_p_error: params.global_p_error,
                is_feasible: true,
                error_msg: String::default(),
            }
        }
    }
}

#[cfg(test)]
mod tests;
