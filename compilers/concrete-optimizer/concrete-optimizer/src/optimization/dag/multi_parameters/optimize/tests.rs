#![allow(clippy::float_cmp)]

use once_cell::sync::Lazy;

use super::*;
use crate::computing_cost::cpu::CpuComplexity;
use crate::config;
use crate::dag::operator::{FunctionTable, LevelledComplexity, Shape};
use crate::dag::unparametrized;
use crate::optimization::dag::solo_key;
use crate::optimization::dag::solo_key::optimize::{add_v0_dag, v0_dag};
use crate::optimization::decomposition;

const CIPHERTEXT_MODULUS_LOG: u32 = 64;
const FFT_PRECISION: u32 = 53;

static SHARED_CACHES: Lazy<PersistDecompCaches> = Lazy::new(|| {
    let processing_unit = config::ProcessingUnit::Cpu;
    decomposition::cache(
        128,
        processing_unit,
        None,
        true,
        CIPHERTEXT_MODULUS_LOG,
        FFT_PRECISION,
    )
});

const _4_SIGMA: f64 = 0.000_063_342_483_999_973;

const LOW_PARTITION: PartitionIndex = PartitionIndex(0);

static CPU_COMPLEXITY: Lazy<CpuComplexity> = Lazy::new(CpuComplexity::default);

fn default_config() -> Config<'static> {
    let complexity_model = Lazy::force(&CPU_COMPLEXITY);
    Config {
        security_level: 128,
        maximum_acceptable_error_probability: _4_SIGMA,
        key_sharing: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
        complexity_model,
    }
}

fn optimize(
    dag: &unparametrized::Dag,
    p_cut: &Option<PartitionCut>,
    default_partition: PartitionIndex,
) -> Option<Parameters> {
    let config = default_config();
    let search_space = SearchSpace::default_cpu();
    super::optimize(
        dag,
        config,
        &search_space,
        &SHARED_CACHES,
        p_cut,
        default_partition,
    )
    .map_or(None, |v| Some(v.1))
}

fn optimize_single(dag: &unparametrized::Dag) -> Option<Parameters> {
    optimize(dag, &Some(PartitionCut::empty()), LOW_PARTITION)
}

fn equiv_single(dag: &unparametrized::Dag) -> Option<bool> {
    let sol_mono = solo_key::optimize::tests::optimize(dag);
    let sol_multi = optimize_single(dag);
    if sol_mono.best_solution.is_none() != sol_multi.is_none() {
        eprintln!("Not same feasibility");
        return Some(false);
    };
    if sol_multi.is_none() {
        return None;
    }
    let equiv =
        sol_mono.best_solution.unwrap().complexity == sol_multi.as_ref().unwrap().complexity;
    if !equiv {
        eprintln!("Not same complexity");
        eprintln!("Single: {:?}", sol_mono.best_solution.unwrap());
        eprintln!("Multi: {:?}", sol_multi.clone().unwrap().complexity);
        eprintln!("Multi: {:?}", sol_multi.unwrap());
    }
    Some(equiv)
}

#[test]
fn optimize_simple_parameter_v0_dag() {
    for precision in 1..11 {
        for manp in 1..25 {
            eprintln!("P M {precision} {manp}");
            let dag = v0_dag(0, precision, manp as f64);
            if let Some(equiv) = equiv_single(&dag) {
                assert!(equiv);
            } else {
                break;
            }
        }
    }
}

#[test]
fn optimize_simple_parameter_rounded_lut_2_layers() {
    for accumulator_precision in 1..11 {
        for precision in 1..accumulator_precision {
            for manp in [1, 8, 16] {
                eprintln!("CASE {accumulator_precision} {precision} {manp}");
                let dag = v0_dag(0, precision, manp as f64);
                if let Some(equiv) = equiv_single(&dag) {
                    assert!(equiv);
                } else {
                    break;
                }
            }
        }
    }
}

fn equiv_2_single(
    dag_multi: &unparametrized::Dag,
    dag_1: &unparametrized::Dag,
    dag_2: &unparametrized::Dag,
) -> Option<bool> {
    let sol_single_1 = solo_key::optimize::tests::optimize(dag_1);
    let sol_single_2 = solo_key::optimize::tests::optimize(dag_2);
    let sol_multi = optimize(dag_multi, &None, LOW_PARTITION);
    let sol_multi_1 = optimize(dag_1, &None, LOW_PARTITION);
    let sol_multi_2 = optimize(dag_2, &None, LOW_PARTITION);
    let feasible_1 = sol_single_1.best_solution.is_some();
    let feasible_2 = sol_single_2.best_solution.is_some();
    let feasible_multi = sol_multi.is_some();
    if (feasible_1 && feasible_2) != feasible_multi {
        eprintln!("Not same feasibility {feasible_1} {feasible_2} {feasible_multi}");
        return Some(false);
    }
    if sol_multi.is_none() {
        return None;
    }
    let sol_multi = sol_multi.unwrap();
    let sol_multi_1 = sol_multi_1.unwrap();
    let sol_multi_2 = sol_multi_2.unwrap();
    let cost_1 = sol_single_1.best_solution.unwrap().complexity;
    let cost_2 = sol_single_2.best_solution.unwrap().complexity;
    let cost_multi = sol_multi.complexity;
    let equiv = cost_1 + cost_2 == cost_multi
        && cost_1 == sol_multi_1.complexity
        && cost_2 == sol_multi_2.complexity
        && sol_multi.micro_params.ks[0][0].unwrap().decomp
            == sol_multi_1.micro_params.ks[0][0].unwrap().decomp
        && sol_multi.micro_params.ks[1][1].unwrap().decomp
            == sol_multi_2.micro_params.ks[0][0].unwrap().decomp;
    if !equiv {
        eprintln!("Not same complexity");
        eprintln!("Multi: {cost_multi:?}");
        eprintln!("Added Single: {:?}", cost_1 + cost_2);
        eprintln!("Single1: {:?}", sol_single_1.best_solution.unwrap());
        eprintln!("Single2: {:?}", sol_single_2.best_solution.unwrap());
        eprintln!("Multi: {sol_multi:?}");
        eprintln!("Multi1: {sol_multi_1:?}");
        eprintln!("Multi2: {sol_multi_2:?}");
    }
    Some(equiv)
}

#[test]
fn optimize_multi_independant_2_precisions() {
    let sum_size = 0;
    for precision1 in 1..11 {
        for precision2 in (precision1 + 1)..11 {
            for manp in [1, 8, 16] {
                eprintln!("CASE {precision1} {precision2} {manp}");
                let noise_factor = manp as f64;
                let mut dag_multi = v0_dag(sum_size, precision1, noise_factor);
                add_v0_dag(&mut dag_multi, sum_size, precision2, noise_factor);
                let dag_1 = v0_dag(sum_size, precision1, noise_factor);
                let dag_2 = v0_dag(sum_size, precision2, noise_factor);
                if let Some(equiv) = equiv_2_single(&dag_multi, &dag_1, &dag_2) {
                    assert!(equiv, "FAILED ON {precision1} {precision2} {manp}");
                } else {
                    break;
                }
            }
        }
    }
}

fn dag_lut_sum_of_2_partitions_2_layer(
    precision1: u8,
    precision2: u8,
    final_lut: bool,
) -> unparametrized::Dag {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(precision1, Shape::number());
    let input2 = dag.add_input(precision2, Shape::number());
    let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, precision1);
    let lut2 = dag.add_lut(input2, FunctionTable::UNKWOWN, precision2);
    let lut1 = dag.add_lut(lut1, FunctionTable::UNKWOWN, precision2);
    let lut2 = dag.add_lut(lut2, FunctionTable::UNKWOWN, precision2);
    let dot = dag.add_dot([lut1, lut2], [1, 1]);
    if final_lut {
        _ = dag.add_lut(dot, FunctionTable::UNKWOWN, precision1);
    }
    dag
}

#[test]
fn optimize_multi_independant_2_partitions_finally_added() {
    let default_partition = 0;
    let single_precision_sol: Vec<_> = (0..11)
        .map(|precision| {
            let dag = dag_lut_sum_of_2_partitions_2_layer(precision, precision, false);
            optimize_single(&dag)
        })
        .collect();

    for precision1 in 1..11 {
        for precision2 in (precision1 + 1)..11 {
            let p_cut = Some(PartitionCut::from_precisions(&[precision1, precision2]));
            let dag_multi = dag_lut_sum_of_2_partitions_2_layer(precision1, precision2, false);
            let sol_1 = single_precision_sol[precision1 as usize].clone();
            let sol_2 = single_precision_sol[precision2 as usize].clone();
            let sol_multi = optimize(&dag_multi, &p_cut, LOW_PARTITION);
            let feasible_multi = sol_multi.is_some();
            let feasible_2 = sol_2.is_some();
            assert!(feasible_multi);
            assert!(feasible_2);
            let sol_multi = sol_multi.unwrap();
            let sol_1 = sol_1.unwrap();
            let sol_2 = sol_2.unwrap();
            assert!(sol_1.complexity < sol_multi.complexity);
            if REAL_FAST_KS {
                assert!(sol_multi.complexity < sol_2.complexity);
            }
            eprintln!("{:?}", sol_multi.micro_params.fks);
            let fks_complexity = sol_multi.micro_params.fks[(default_partition + 1) % 2]
                [default_partition]
                .unwrap()
                .complexity;
            let sol_multi_without_fks = sol_multi.complexity - fks_complexity;
            let perfect_complexity = (sol_1.complexity + sol_2.complexity) / 2.0;
            if REAL_FAST_KS {
                assert!(sol_multi.macro_params[1] == sol_2.macro_params[0]);
            }
            // The smallest the precision the more fks noise break partition independence
            #[allow(clippy::collapsible_else_if)]
            let maximal_relative_degratdation = if REAL_FAST_KS {
                if precision1 < 4 {
                    1.1
                } else if precision1 <= 7 {
                    1.03
                } else {
                    1.001
                }
            } else {
                if precision1 < 4 {
                    1.5
                } else if precision1 <= 7 {
                    1.8
                } else {
                    1.6
                }
            };
            assert!(
                sol_multi_without_fks / perfect_complexity < maximal_relative_degratdation,
                "{precision1} {precision2} {} < {maximal_relative_degratdation}",
                sol_multi_without_fks / perfect_complexity
            );
        }
    }
}

#[test]
fn optimize_multi_independant_2_partitions_finally_added_and_luted() {
    let default_partition = 0;
    let single_precision_sol: Vec<_> = (0..11)
        .map(|precision| {
            let dag = dag_lut_sum_of_2_partitions_2_layer(precision, precision, true);
            optimize_single(&dag)
        })
        .collect();
    for precision1 in 1..11 {
        for precision2 in (precision1 + 1)..11 {
            let p_cut = Some(PartitionCut::from_precisions(&[precision1, precision2]));
            let dag_multi = dag_lut_sum_of_2_partitions_2_layer(precision1, precision2, true);
            let sol_1 = single_precision_sol[precision1 as usize].clone();
            let sol_2 = single_precision_sol[precision2 as usize].clone();
            let sol_multi = optimize(&dag_multi, &p_cut, PartitionIndex(0));
            let feasible_multi = sol_multi.is_some();
            let feasible_2 = sol_2.is_some();
            assert!(feasible_multi);
            assert!(feasible_2);
            let sol_multi = sol_multi.unwrap();
            let sol_1 = sol_1.unwrap();
            let sol_2 = sol_2.unwrap();
            // The smallest the precision the more fks noise dominate
            assert!(sol_1.complexity < sol_multi.complexity);
            assert!(sol_multi.complexity < sol_2.complexity);
            let fks_complexity = sol_multi.micro_params.fks[(default_partition + 1) % 2]
                [default_partition]
                .unwrap()
                .complexity;
            let sol_multi_without_fks = sol_multi.complexity - fks_complexity;
            let perfect_complexity = (sol_1.complexity + sol_2.complexity) / 2.0;
            let relative_degradation = sol_multi_without_fks / perfect_complexity;
            #[allow(clippy::collapsible_else_if)]
            let maxim_relative_degradation = if REAL_FAST_KS {
                if precision1 < 4 {
                    1.2
                } else if precision1 <= 7 {
                    1.19
                } else {
                    1.15
                }
            } else {
                if precision1 < 4 {
                    1.45
                } else if precision1 <= 7 {
                    1.8
                } else {
                    1.6
                }
            };
            assert!(
                relative_degradation < maxim_relative_degradation,
                "{precision1} {precision2} {}",
                sol_multi_without_fks / perfect_complexity
            );
        }
    }
}

fn optimize_rounded(dag: &unparametrized::Dag) -> Option<Parameters> {
    let p_cut = Some(PartitionCut::from_precisions(&[1, 128]));
    let default_partition = PartitionIndex(0);
    optimize(dag, &p_cut, default_partition)
}

fn dag_rounded_lut_2_layers(accumulator_precision: usize, precision: usize) -> unparametrized::Dag {
    let out_precision = accumulator_precision as u8;
    let rounded_precision = precision as u8;
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(precision as u8, Shape::number());
    let rounded1 = dag.add_expanded_rounded_lut(
        input1,
        FunctionTable::UNKWOWN,
        rounded_precision,
        out_precision,
    );
    let rounded2 = dag.add_expanded_rounded_lut(
        rounded1,
        FunctionTable::UNKWOWN,
        rounded_precision,
        out_precision,
    );
    let _rounded3 =
        dag.add_expanded_rounded_lut(rounded2, FunctionTable::UNKWOWN, rounded_precision, 1);
    dag
}

fn test_optimize_v3_expanded_round(
    precision_acc: usize,
    precision_tlu: usize,
    minimal_speedup: f64,
) {
    let dag = dag_rounded_lut_2_layers(precision_acc, precision_tlu);
    let sol_mono = solo_key::optimize::tests::optimize(&dag)
        .best_solution
        .unwrap();
    let sol = optimize_rounded(&dag).unwrap();
    let speedup = sol_mono.complexity / sol.complexity;
    assert!(
        speedup >= minimal_speedup,
        "Speedup {speedup} smaller than {minimal_speedup} for {precision_acc}/{precision_tlu}"
    );
    let expected_ks = [
        [true, true],  // KS[0], KS[0->1]
        [false, true], // KS[1]
    ];
    let expected_fks = [
        [false, false],
        [true, false], // FKS[1->0]
    ];
    for (src, dst) in cross_partition(2) {
        assert!(sol.micro_params.ks[src.0][dst.0].is_some() == expected_ks[src.0][dst.0]);
        assert!(sol.micro_params.fks[src.0][dst.0].is_some() == expected_fks[src.0][dst.0]);
    }
}

#[test]
fn test_optimize_v3_expanded_round_16_8() {
    if REAL_FAST_KS {
        test_optimize_v3_expanded_round(16, 8, 5.5);
    } else {
        test_optimize_v3_expanded_round(16, 8, 3.9);
    }
}

#[test]
fn test_optimize_v3_expanded_round_16_6() {
    if REAL_FAST_KS {
        test_optimize_v3_expanded_round(16, 6, 3.3);
    } else {
        test_optimize_v3_expanded_round(16, 6, 2.6);
    }
}

#[test]
fn optimize_v3_direct_round() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(16, Shape::number());
    _ = dag.add_expanded_rounded_lut(input1, FunctionTable::UNKWOWN, 8, 16);
    let sol = optimize_rounded(&dag).unwrap();
    let sol_mono = solo_key::optimize::tests::optimize(&dag)
        .best_solution
        .unwrap();
    let minimal_speedup = 8.6;
    let speedup = sol_mono.complexity / sol.complexity;
    assert!(
        speedup >= minimal_speedup,
        "Speedup {speedup} smaller than {minimal_speedup}"
    );
}

#[test]
fn optimize_sign_extract() {
    let precision = 8;
    let high_precision = 16;
    let mut dag = unparametrized::Dag::new();
    let complexity = LevelledComplexity::ZERO;
    let free_small_input1 = dag.add_input(precision, Shape::number());
    let small_input1 = dag.add_lut(free_small_input1, FunctionTable::UNKWOWN, precision);
    let small_input1 = dag.add_lut(small_input1, FunctionTable::UNKWOWN, high_precision);
    let input1 = dag.add_levelled_op(
        [small_input1],
        complexity,
        1.0,
        Shape::vector(1_000_000),
        "comment",
    );
    let rounded1 = dag.add_expanded_round(input1, 1);
    let _rounded2 = dag.add_lut(rounded1, FunctionTable::UNKWOWN, 1);
    let sol = optimize_rounded(&dag).unwrap();
    let sol_mono = solo_key::optimize::tests::optimize(&dag)
        .best_solution
        .unwrap();
    let speedup = sol_mono.complexity / sol.complexity;
    let minimal_speedup = if REAL_FAST_KS { 80.0 } else { 30.0 };
    assert!(
        speedup >= minimal_speedup,
        "Speedup {speedup} smaller than {minimal_speedup}"
    );
}

fn test_partition_chain(decreasing: bool) {
    // tlu chain with decreasing precision (decreasing partition index)
    // check that increasing partitionning gaves faster solutions
    // check solution has the right structure
    let mut dag = unparametrized::Dag::new();
    let min_precision = 6;
    let max_precision = 8;
    let mut input_precisions: Vec<_> = (min_precision..=max_precision).collect();
    if decreasing {
        input_precisions.reverse();
    }
    let mut lut_input = dag.add_input(input_precisions[0], Shape::number());
    for &out_precision in &input_precisions {
        lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
    }
    lut_input = dag.add_lut(
        lut_input,
        FunctionTable::UNKWOWN,
        *input_precisions.last().unwrap(),
    );
    _ = dag.add_lut(lut_input, FunctionTable::UNKWOWN, min_precision);
    let mut p_cut = PartitionCut::empty();
    let sol = optimize(&dag, &Some(p_cut.clone()), PartitionIndex(0)).unwrap();
    assert!(sol.macro_params.len() == 1);
    let mut complexity = sol.complexity;
    for &out_precision in &input_precisions {
        if out_precision == max_precision {
            // There is nothing to cut above max_precision
            continue;
        }
        p_cut.p_cut.push((out_precision, f64::MAX));
        p_cut.p_cut.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sol = optimize(&dag, &Some(p_cut.clone()), PartitionIndex(0)).unwrap();
        let nb_partitions = sol.macro_params.len();
        assert!(
            nb_partitions == (p_cut.p_cut.len() + 1),
            "bad nb partitions {} {p_cut}",
            sol.macro_params.len()
        );
        assert!(
            sol.complexity < complexity,
            "{} < {complexity} {out_precision} / {max_precision}",
            sol.complexity
        );
        for (src, dst) in cross_partition(nb_partitions) {
            let ks = sol.micro_params.ks[src.0][dst.0];
            eprintln!("{} {src} {dst}", ks.is_some());
            let expected_ks = (!decreasing || src.0 == dst.0 + 1)
                && (decreasing || src.0 + 1 == dst.0)
                || (src == dst && (src == PartitionIndex(0) || src.0 == nb_partitions - 1));
            assert!(
                ks.is_some() == expected_ks,
                "{:?} {:?}",
                ks.is_some(),
                expected_ks
            );
            let fks = sol.micro_params.fks[src.0][dst.0];
            assert!(fks.is_none());
        }
        complexity = sol.complexity;
    }
    let sol = optimize(&dag, &None, PartitionIndex(0));
    assert!(sol.unwrap().complexity == complexity);
}

#[test]
fn test_partition_decreasing_chain() {
    test_partition_chain(true);
}

#[test]
fn test_partition_increasing_chain() {
    test_partition_chain(true);
}

const MAX_WEIGHT: &[u64] = &[
    // max v0 weight for each precision
    1_073_741_824,
    1_073_741_824, // 2**30, 1b
    536_870_912,   // 2**29, 2b
    268_435_456,   // 2**28, 3b
    67_108_864,    // 2**26, 4b
    16_777_216,    // 2**24, 5b
    4_194_304,     // 2**22, 6b
    1_048_576,     // 2**20, 7b
    262_144,       // 2**18, 8b
    65_536,        // 2**16, 9b
    16384,         // 2**14, 10b
    2048,          // 2**11, 11b
];

#[test]
fn test_independant_partitions_non_feasible_single_params() {
    // generate hard circuit, non feasible with single parameters
    // composed of independant partitions so we know the optimal result
    let precisions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let sum_size = 0;
    let noise_factor = MAX_WEIGHT[precisions[0]] as f64;
    let mut dag = v0_dag(sum_size, precisions[0] as u64, noise_factor);
    let sol_single = optimize_single(&dag);
    let mut optimal_complexity = sol_single.as_ref().unwrap().complexity;
    let mut optimal_p_error = sol_single.unwrap().p_error;
    for &out_precision in &precisions[1..] {
        let noise_factor = MAX_WEIGHT[out_precision] as f64;
        add_v0_dag(&mut dag, sum_size, out_precision as u64, noise_factor);
        let sol_single = optimize_single(&v0_dag(sum_size, out_precision as u64, noise_factor));
        optimal_complexity += sol_single.as_ref().unwrap().complexity;
        optimal_p_error += sol_single.as_ref().unwrap().p_error;
    }
    // check non feasible in single
    let sol_single = solo_key::optimize::tests::optimize(&dag).best_solution;
    assert!(sol_single.is_none());
    // solves in multi
    let sol = optimize(&dag, &None, PartitionIndex(0));
    assert!(sol.is_some());
    let sol = sol.unwrap();
    // check optimality
    assert!(sol.complexity / optimal_complexity < 1.0 + f64::EPSILON);
    assert!(sol.p_error / optimal_p_error < 1.0 + f64::EPSILON);
}

#[test]
fn test_chained_partitions_non_feasible_single_params() {
    // generate hard circuit, non feasible with single parameters
    let precisions = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    // Note: reversing chain have issues for connecting lower bits to 7 bits, there may be no feasible solution
    let mut dag = unparametrized::Dag::new();
    let mut lut_input = dag.add_input(precisions[0], Shape::number());
    for out_precision in precisions {
        let noise_factor = MAX_WEIGHT[*dag.out_precisions.last().unwrap() as usize] as f64;
        lut_input = dag.add_levelled_op(
            [lut_input],
            LevelledComplexity::ZERO,
            noise_factor,
            Shape::number(),
            "",
        );
        lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
    }
    _ = dag.add_lut(
        lut_input,
        FunctionTable::UNKWOWN,
        *precisions.last().unwrap(),
    );
    let sol_single = solo_key::optimize::tests::optimize(&dag).best_solution;
    assert!(sol_single.is_none());
    let sol = optimize(&dag, &None, PartitionIndex(0));
    assert!(sol.is_some());
}

#[test]
fn test_multi_rounded_fks_coherency() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(16, Shape::number());
    let reduced_8 = dag.add_expanded_rounded_lut(input1, FunctionTable::UNKWOWN, 8, 8);
    let reduced_4 = dag.add_expanded_rounded_lut(input1, FunctionTable::UNKWOWN, 4, 8);
    _ = dag.add_dot([reduced_8, reduced_4], [1, 1]);
    let sol = optimize(&dag, &None, PartitionIndex(0));
    assert!(sol.is_some());
    let sol = sol.unwrap();
    for (src, dst) in cross_partition(sol.macro_params.len()) {
        if let Some(fks) = sol.micro_params.fks[src.0][dst.0] {
            assert!(fks.src_glwe_param == sol.macro_params[src.0].unwrap().glwe_params);
            assert!(fks.dst_glwe_param == sol.macro_params[dst.0].unwrap().glwe_params);
        }
    }
}

#[test]
fn test_levelled_only() {
    let mut dag = unparametrized::Dag::new();
    let _ = dag.add_input(22, Shape::number());
    let config = default_config();
    let search_space = SearchSpace::default_cpu();
    let sol =
        super::optimize_to_circuit_solution(&dag, config, &search_space, &SHARED_CACHES, &None);
    let sol_mono = solo_key::optimize::tests::optimize(&dag)
        .best_solution
        .unwrap();
    assert!(sol.circuit_keys.secret_keys.len() == 1);
    assert!(sol.circuit_keys.secret_keys[0].polynomial_size == sol_mono.glwe_polynomial_size);
}

#[test]
fn test_big_secret_key_sharing() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(4, Shape::number());
    let input2 = dag.add_input(5, Shape::number());
    let input2 = dag.add_dot([input2], [128]);
    let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 5);
    let lut2 = dag.add_lut(input2, FunctionTable::UNKWOWN, 5);
    let _ = dag.add_dot([lut1, lut2], [16, 1]);
    let config_sharing = Config {
        security_level: 128,
        maximum_acceptable_error_probability: _4_SIGMA,
        key_sharing: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
        complexity_model: &CpuComplexity::default(),
    };
    let config_no_sharing = Config {
        key_sharing: false,
        ..config_sharing
    };
    let mut search_space = SearchSpace::default_cpu();
    // eprintln!("{:?}", search_space);
    search_space.glwe_dimensions = vec![1]; // forcing big key sharing
    let sol_sharing = super::optimize_to_circuit_solution(
        &dag,
        config_sharing,
        &search_space,
        &SHARED_CACHES,
        &None,
    );
    eprintln!("NO SHARING");
    let sol_no_sharing = super::optimize_to_circuit_solution(
        &dag,
        config_no_sharing,
        &search_space,
        &SHARED_CACHES,
        &None,
    );
    let keys_sharing = sol_sharing.circuit_keys;
    let keys_no_sharing = sol_no_sharing.circuit_keys;
    assert!(keys_sharing.secret_keys.len() == 3);
    assert!(keys_no_sharing.secret_keys.len() == 4);
    assert!(keys_sharing.conversion_keyswitch_keys.is_empty());
    assert!(keys_no_sharing.conversion_keyswitch_keys.len() == 1);
    assert!(keys_sharing.bootstrap_keys.len() == keys_no_sharing.bootstrap_keys.len());
    assert!(keys_sharing.keyswitch_keys.len() == keys_no_sharing.keyswitch_keys.len());
}

#[test]
fn test_big_and_small_secret_key() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(4, Shape::number());
    let input2 = dag.add_input(5, Shape::number());
    let input2 = dag.add_dot([input2], [128]);
    let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 5);
    let lut2 = dag.add_lut(input2, FunctionTable::UNKWOWN, 5);
    let _ = dag.add_dot([lut1, lut2], [16, 1]);
    let config_sharing = Config {
        security_level: 128,
        maximum_acceptable_error_probability: _4_SIGMA,
        key_sharing: true,
        ciphertext_modulus_log: 64,
        fft_precision: 53,
        complexity_model: &CpuComplexity::default(),
    };
    let config_no_sharing = Config {
        key_sharing: false,
        ..config_sharing
    };
    let mut search_space = SearchSpace::default_cpu();
    search_space.glwe_dimensions = vec![1]; // forcing big key sharing
    search_space.internal_lwe_dimensions = vec![768]; // forcing small key sharing
    let sol_sharing = super::optimize_to_circuit_solution(
        &dag,
        config_sharing,
        &search_space,
        &SHARED_CACHES,
        &None,
    );
    let sol_no_sharing = super::optimize_to_circuit_solution(
        &dag,
        config_no_sharing,
        &search_space,
        &SHARED_CACHES,
        &None,
    );
    let keys_sharing = sol_sharing.circuit_keys;
    let keys_no_sharing = sol_no_sharing.circuit_keys;
    assert!(keys_sharing.secret_keys.len() == 2);
    assert!(keys_no_sharing.secret_keys.len() == 4);
    assert!(keys_sharing.conversion_keyswitch_keys.is_empty());
    assert!(keys_no_sharing.conversion_keyswitch_keys.len() == 1);
    // boostrap are merged due to same (level, base)
    assert!(keys_sharing.bootstrap_keys.len() + 1 == keys_no_sharing.bootstrap_keys.len());
    // keyswitch are still different due to another (level, base)
    assert!(keys_sharing.keyswitch_keys.len() == keys_no_sharing.keyswitch_keys.len());
}

#[test]
fn test_composition_2_partitions() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(3, Shape::number());
    let lut1 = dag.add_lut(input1, FunctionTable::UNKWOWN, 6);
    let lut3 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 3);
    let input2 = dag.add_dot([input1, lut3], [1, 1]);
    let out = dag.add_lut(input2, FunctionTable::UNKWOWN, 3);
    let search_space = SearchSpace::default_cpu();
    let normal_sol = super::optimize(
        &dag,
        default_config(),
        &search_space,
        &SHARED_CACHES,
        &None,
        PartitionIndex(1),
    )
    .unwrap()
    .1;
    dag.add_composition(out, input1);
    let composed_sol = super::optimize(
        &dag,
        default_config(),
        &search_space,
        &SHARED_CACHES,
        &None,
        PartitionIndex(1),
    )
    .unwrap()
    .1;
    assert!(composed_sol.is_feasible);
    assert!(composed_sol.complexity > normal_sol.complexity);
}

#[test]
fn test_composition_1_partition_not_composable() {
    let mut dag = unparametrized::Dag::new();
    let input1 = dag.add_input(8, Shape::number());
    let dot = dag.add_dot([input1], [1 << 16]);
    let lut1 = dag.add_lut(dot, FunctionTable::UNKWOWN, 8);
    let oup = dag.add_dot([lut1], [1 << 16]);
    let normal_config = default_config();
    let composed_config = normal_config;
    let search_space = SearchSpace::default_cpu();
    let normal_sol = super::optimize(
        &dag,
        normal_config,
        &search_space,
        &SHARED_CACHES,
        &None,
        PartitionIndex(1),
    );
    dag.add_composition(oup, input1);
    let composed_sol = super::optimize(
        &dag,
        composed_config,
        &search_space,
        &SHARED_CACHES,
        &None,
        PartitionIndex(1),
    );
    assert!(normal_sol.is_ok());
    assert!(composed_sol.is_err());
}

#[test]
fn test_maximal_multi() {
    let config = default_config();
    let search_space = SearchSpace::default_cpu();
    let mut dag = unparametrized::Dag::new();
    let input = dag.add_input(8, Shape::number());
    let lut1 = dag.add_lut(input, FunctionTable::UNKWOWN, 8u8);
    let lut2 = dag.add_lut(lut1, FunctionTable::UNKWOWN, 8u8);
    _ = dag.add_dot([lut2], [1 << 16]);

    let sol = optimize(&dag, &None, PartitionIndex(0)).unwrap();
    assert!(sol.macro_params.len() == 1);

    let p_cut = PartitionCut::maximal_partitionning(&dag);
    let sol = optimize(&dag, &Some(p_cut.clone()), PartitionIndex(0)).unwrap();
    assert!(sol.macro_params.len() == 2);

    eprintln!("{:?}", sol.micro_params.pbs);

    let sol_ref =
        super::optimize_to_circuit_solution(&dag, config, &search_space, &SHARED_CACHES, &None);
    assert!(sol_ref.circuit_keys.secret_keys.len() == 2);

    let sol = super::optimize_to_circuit_solution(
        &dag,
        config,
        &search_space,
        &SHARED_CACHES,
        &Some(p_cut),
    );
    assert!(sol.circuit_keys.secret_keys.len() == 3);
    let expected_speedup = (2 * 5859) as f64 / (1721 + 5859) as f64; // from ./optimizer i.e. v0
    eprintln!("{} vs {}", sol.complexity, sol_ref.complexity);
    // note: we have a 5% relative margin since dag complexity is slightly better than v0
    assert!(sol.complexity < 1.05 * (sol_ref.complexity / expected_speedup));
}

#[test]
fn test_bug_with_zero_noise() {
    let complexity = LevelledComplexity::ZERO;
    let out_shape = Shape::number();
    let mut dag = unparametrized::Dag::new();
    let v0 = dag.add_input(2, &out_shape);
    let v1 = dag.add_levelled_op([v0], complexity, 0.0, &out_shape, "comment");
    let v2 = dag.add_levelled_op([v1], complexity, 1.0, &out_shape, "comment");
    let v3 = dag.add_unsafe_cast(v2, 1);
    let _ = dag.add_lut(v3, FunctionTable { values: vec![] }, 1);
    let sol = optimize(&dag, &None, PartitionIndex(0));
    assert!(sol.is_some());
}
