// part of optimize.rs
#[cfg(test)]
mod tests {
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

    static SHARED_CACHES: Lazy<PersistDecompCaches> = Lazy::new(|| {
        let processing_unit = config::ProcessingUnit::Cpu;
        decomposition::cache(128, processing_unit, None, true)
    });

    const _4_SIGMA: f64 = 0.000_063_342_483_999_973;

    const LOW_PARTITION: PartitionIndex = 0;

    fn optimize(
        dag: &unparametrized::OperationDag,
        p_cut: &Option<PrecisionCut>,
        default_partition: usize,
    ) -> Option<Parameters> {
        let config = Config {
            security_level: 128,
            maximum_acceptable_error_probability: _4_SIGMA,
            ciphertext_modulus_log: 64,
            complexity_model: &CpuComplexity::default(),
        };

        let search_space = SearchSpace::default_cpu();
        super::optimize(
            dag,
            config,
            &search_space,
            &SHARED_CACHES,
            p_cut,
            default_partition,
        )
    }

    fn optimize_single(dag: &unparametrized::OperationDag) -> Option<Parameters> {
        optimize(dag, &Some(PrecisionCut { p_cut: vec![] }), LOW_PARTITION)
    }

    fn equiv_single(dag: &unparametrized::OperationDag) -> Option<bool> {
        let sol_mono = solo_key::optimize::tests::optimize(dag);
        let sol_multi = optimize_single(dag);
        if sol_mono.best_solution.is_none() != sol_multi.is_none() {
            eprintln!("Not same feasibility");
            return Some(false);
        };
        if sol_multi.is_none() {
            return None;
        }
        let equiv = sol_mono.best_solution.unwrap().complexity
            == sol_multi.as_ref().unwrap().complexity;
        if !equiv {
            eprintln!("Not same complexity");
            eprintln!("Single: {:?}", sol_mono.best_solution.unwrap());
            eprintln!(
                "Multi: {:?}",
                sol_multi.clone().unwrap().complexity
            );
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
        dag_multi: &unparametrized::OperationDag,
        dag_1: &unparametrized::OperationDag,
        dag_2: &unparametrized::OperationDag,
    ) -> Option<bool> {
        let precision_max = dag_multi.out_precisions.iter().copied().max().unwrap();
        let p_cut = Some(PrecisionCut {
            p_cut: vec![precision_max - 1],
        });
        eprintln!("{dag_multi}");
        let sol_single_1 = solo_key::optimize::tests::optimize(dag_1);
        let sol_single_2 = solo_key::optimize::tests::optimize(dag_2);
        let sol_multi = optimize(dag_multi, &p_cut, LOW_PARTITION);
        let sol_multi_1 = optimize(dag_1, &p_cut, LOW_PARTITION);
        let sol_multi_2 = optimize(dag_2, &p_cut, LOW_PARTITION);
        let feasible_1 = sol_single_1.best_solution.is_some();
        let feasible_2 = sol_single_2.best_solution.is_some();
        let feasible_multi = sol_multi.is_some();
        if (feasible_1 && feasible_2) != feasible_multi {
            eprintln!(
                "Not same feasibility {feasible_1} {feasible_2} {feasible_multi}"
            );
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
        let equiv =
            cost_1 + cost_2 == cost_multi
            && cost_1 == sol_multi_1.complexity
            && cost_2 == sol_multi_2.complexity
            && sol_multi.micro_params.ks[0][0].unwrap().decomp ==
            sol_multi_1.micro_params.ks[0][0].unwrap().decomp
            && sol_multi.micro_params.ks[1][1].unwrap().decomp ==
            sol_multi_2.micro_params.ks[0][0].unwrap().decomp
        ;
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

    fn dag_lut_sum_of_2_partitions_2_layer(precision1: u8, precision2: u8, final_lut: bool) -> unparametrized::OperationDag {
        let mut dag = unparametrized::OperationDag::new();
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
        let single_precision_sol : Vec<_> = (0..11).map(
            |precision| {
                let dag = dag_lut_sum_of_2_partitions_2_layer(precision, precision, false);
                optimize_single(&dag)
            }
        ).collect();

        for precision1 in 1..11 {
            for precision2 in (precision1 + 1)..11 {
                let p_cut = Some(PrecisionCut {
                    p_cut: vec![precision1],
                });
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
                assert!(sol_multi.complexity < sol_2.complexity);
                eprintln!("{:?}", sol_multi.micro_params.fks);
                let fks_complexity = sol_multi.micro_params.fks[(default_partition + 1) % 2][default_partition].unwrap().complexity;
                let sol_multi_without_fks = sol_multi.complexity - fks_complexity;
                let perfect_complexity = (sol_1.complexity + sol_2.complexity) / 2.0;
                assert!(sol_multi.macro_params[1] == sol_2.macro_params[0]);
                // The smallest the precision the more fks noise break partition independence
                if precision1 < 4 {
                    assert!(
                        sol_multi_without_fks / perfect_complexity < 1.1,
                        "{precision1} {precision2}"
                    );
                } else if precision1 <= 7 {
                    assert!(
                        sol_multi_without_fks / perfect_complexity < 1.03,
                        "{precision1} {precision2} {}", sol_multi_without_fks / perfect_complexity
                    );
                } else {
                    assert!(
                        sol_multi_without_fks / perfect_complexity < 1.001,
                        "{precision1} {precision2} {}", sol_multi_without_fks / perfect_complexity
                    );
                }
            }
        }
    }

    #[test]
    fn optimize_multi_independant_2_partitions_finally_added_and_luted() {
        let default_partition = 0;
        let single_precision_sol : Vec<_> = (0..11).map(
            |precision| {
                let dag = dag_lut_sum_of_2_partitions_2_layer(precision, precision, true);
                optimize_single(&dag)
            }
        ).collect();
        for precision1 in 1..11 {
            for precision2 in (precision1 + 1)..11 {
                let p_cut = Some(PrecisionCut {
                    p_cut: vec![precision1],
                });
                let dag_multi = dag_lut_sum_of_2_partitions_2_layer(precision1, precision2, true);
                let sol_1 = single_precision_sol[precision1 as usize].clone();
                let sol_2 = single_precision_sol[precision2 as usize].clone();
                let sol_multi = optimize(&dag_multi, &p_cut, 0);
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
                let fks_complexity = sol_multi.micro_params.fks[(default_partition + 1) % 2][default_partition].unwrap().complexity;
                let sol_multi_without_fks = sol_multi.complexity - fks_complexity;
                let perfect_complexity = (sol_1.complexity + sol_2.complexity) / 2.0;
                let relative_degradation = sol_multi_without_fks / perfect_complexity;
                if precision1 < 4 {
                    assert!(
                        relative_degradation < 1.2,
                        "{precision1} {precision2} {}", sol_multi_without_fks / perfect_complexity
                    );
                } else if precision1 <= 7 {
                    assert!(
                        relative_degradation < 1.19,
                        "{precision1} {precision2} {}", sol_multi_without_fks / perfect_complexity
                    );
                } else {
                    assert!(
                        relative_degradation < 1.15,
                        "{precision1} {precision2} {}", sol_multi_without_fks / perfect_complexity
                    );
                }
            }
        }
    }

    fn optimize_rounded(dag: &unparametrized::OperationDag) -> Option<Parameters> {
        let p_cut = Some(PrecisionCut { p_cut: vec![1] });
        let default_partition = 0;
        optimize(dag, &p_cut, default_partition)
    }

    fn dag_rounded_lut_2_layers(
        accumulator_precision: usize,
        precision: usize,
    ) -> unparametrized::OperationDag {
        let out_precision = accumulator_precision as u8;
        let rounded_precision = precision as u8;
        let mut dag = unparametrized::OperationDag::new();
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

    fn test_optimize_v3_expanded_round(precision_acc: usize, precision_tlu: usize, minimal_speedup: f64) {
        let dag = dag_rounded_lut_2_layers(precision_acc, precision_tlu);
        let sol_mono = solo_key::optimize::tests::optimize(&dag).best_solution.unwrap();
        let sol = optimize_rounded(&dag).unwrap();
        let speedup = sol_mono.complexity / sol.complexity;
        assert!(speedup >= minimal_speedup,
            "Speedup {speedup} smaller than {minimal_speedup} for {precision_acc}/{precision_tlu}"
        );
        let expected_ks = [
            [true, true], // KS[0], KS[0->1]
            [false, true],// KS[1]
        ];
        let expected_fks = [
            [false, false],
            [true, false], // FKS[1->0]
        ];
        for (src, dst) in cross_partition(2) {
            assert!(sol.micro_params.ks[src][dst].is_some() == expected_ks[src][dst]);
            assert!(sol.micro_params.fks[src][dst].is_some() == expected_fks[src][dst]);
        }
    }

    #[test]
    fn test_optimize_v3_expanded_round_16_8() {
        test_optimize_v3_expanded_round(16, 8, 5.5);
    }

    #[test]
    fn test_optimize_v3_expanded_round_16_6() {
        test_optimize_v3_expanded_round(16, 6, 3.3);
    }

    #[test]
    fn optimize_v3_direct_round() {
        let mut dag = unparametrized::OperationDag::new();
        let input1 = dag.add_input(16, Shape::number());
        _ = dag.add_expanded_rounded_lut(input1, FunctionTable::UNKWOWN, 8, 16);
        let sol = optimize_rounded(&dag).unwrap();
        let sol_mono = solo_key::optimize::tests::optimize(&dag).best_solution.unwrap();
        let minimal_speedup = 8.6;
        let speedup = sol_mono.complexity / sol.complexity;
        assert!(speedup >= minimal_speedup,
            "Speedup {speedup} smaller than {minimal_speedup}"
        );
    }

    #[test]
    fn optimize_sign_extract() {
        let precision = 8;
        let high_precision = 16;
        let mut dag = unparametrized::OperationDag::new();
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
        let sol_mono = solo_key::optimize::tests::optimize(&dag).best_solution.unwrap();
        let speedup = sol_mono.complexity / sol.complexity;
        let minimal_speedup = 80.0;
        assert!(speedup >= minimal_speedup,
            "Speedup {speedup} smaller than {minimal_speedup}"
        );
    }

    fn test_partition_chain(decreasing: bool) {
        // tlu chain with decreasing precision (decreasing partition index)
        // check that increasing partitionning gaves faster solutions
        // check solution has the right structure
        let mut dag = unparametrized::OperationDag::new();
        let min_precision = 6;
        let max_precision = 8;
        let mut input_precisions : Vec<_> = (min_precision..=max_precision).collect();
        if decreasing {
            input_precisions.reverse();
        }
        let mut lut_input = dag.add_input(input_precisions[0], Shape::number());
        for &out_precision in &input_precisions {
            lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
        }
        lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, *input_precisions.last().unwrap());
        _ = dag.add_lut(lut_input, FunctionTable::UNKWOWN, min_precision);
        let mut p_cut = PrecisionCut { p_cut:vec![] };
        let sol = optimize(&dag, &Some(p_cut.clone()), 0).unwrap();
        assert!(sol.macro_params.len() == 1);
        let mut complexity = sol.complexity;
        for &out_precision in &input_precisions {
            if out_precision == max_precision {
                // There is nothing to cut above max_precision
                continue;
            }
            p_cut.p_cut.push(out_precision);
            p_cut.p_cut.sort_unstable();
            eprintln!("PCUT {p_cut}");
            let sol = optimize(&dag, &Some(p_cut.clone()), 0).unwrap();
            let nb_partitions = sol.macro_params.len();
            assert!(nb_partitions == (p_cut.p_cut.len() + 1),
                "bad nb partitions {} {p_cut}", sol.macro_params.len());
            assert!(sol.complexity < complexity,
                "{} < {complexity} {out_precision} / {max_precision}", sol.complexity);
            for (src, dst) in cross_partition(nb_partitions) {
                let ks = sol.micro_params.ks[src][dst];
                eprintln!("{} {src} {dst}", ks.is_some());
                let expected_ks =
                    (!decreasing || src == dst + 1)
                    && (decreasing || src + 1 == dst)
                    || (src == dst && (src == 0 || src == nb_partitions - 1))
                ;
                assert!(ks.is_some() == expected_ks, "{:?} {:?}", ks.is_some(), expected_ks);
                let fks = sol.micro_params.fks[src][dst];
                assert!(fks.is_none());
            }
            complexity = sol.complexity;
        }
        let sol = optimize(&dag, &None, 0);
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
        536_870_912,  // 2**29, 2b
        268_435_456,  // 2**28, 3b
        67_108_864,   // 2**26, 4b
        16_777_216,   // 2**24, 5b
        4_194_304,    // 2**22, 6b
        1_048_576,    // 2**20, 7b
        262_144,     // 2**18, 8b
        65_536,      // 2**16, 9b
        16384,      // 2**14, 10b
        2048,       // 2**11, 11b
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
            let noise_factor =  MAX_WEIGHT[out_precision] as f64;
            add_v0_dag(&mut dag, sum_size, out_precision as u64, noise_factor);
            let sol_single = optimize_single(&v0_dag(sum_size, out_precision as u64, noise_factor));
            optimal_complexity += sol_single.as_ref().unwrap().complexity;
            optimal_p_error += sol_single.as_ref().unwrap().p_error;
        }
        // check non feasible in single
        let sol_single = solo_key::optimize::tests::optimize(&dag).best_solution;
        assert!(sol_single.is_none());
        // solves in multi
        let sol = optimize(&dag, &None, 0);
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
        let mut dag = unparametrized::OperationDag::new();
        let mut lut_input = dag.add_input(precisions[0], Shape::number());
        for out_precision in precisions {
            let noise_factor = MAX_WEIGHT[*dag.out_precisions.last().unwrap() as usize] as f64;
            lut_input = dag.add_levelled_op([lut_input], LevelledComplexity::ZERO, noise_factor, Shape::number(), "");
            lut_input = dag.add_lut(lut_input, FunctionTable::UNKWOWN, out_precision);
        }
        _ = dag.add_lut(lut_input, FunctionTable::UNKWOWN, *precisions.last().unwrap());
        let sol_single = solo_key::optimize::tests::optimize(&dag).best_solution;
        assert!(sol_single.is_none());
        let sol = optimize(&dag, &None, 0);
        assert!(sol.is_some());
    }
}
