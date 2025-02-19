use crate::{
    config::ProcessingUnit,
    dag::{
        operator::{FunctionTable, LevelledComplexity, Precision, Shape},
        unparametrized,
    },
    optimization::{
        config::{Config, SearchSpace},
        decomposition::{self},
    },
};

use super::{
    keys_spec::{CircuitKeys, ExpandedCircuitKeys},
    optimize::{optimize, NoSearchSpaceRestriction},
    partition_cut::{ExternalPartition, PartitionCut},
    PartitionIndex,
};

const _4_SIGMA: f64 = 0.000_063_342_483_999_973;

#[derive(Debug, Clone, PartialEq)]
pub struct InternalPartition {
    pub precision: Precision,
    pub norm2: f64,
}

impl PartialOrd for InternalPartition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.precision.cmp(&other.precision) {
            std::cmp::Ordering::Equal => self.norm2.partial_cmp(&other.norm2),
            ordering => Some(ordering),
        }
    }
}

/// Adds subgraphs to the dag, to force the generation of pbs and ks parameters.
fn build_internal_pbs_ks(dag: &mut unparametrized::Dag, internal_partitions: &[InternalPartition]) {
    for def_a in internal_partitions.iter() {
        for def_b in internal_partitions.iter() {
            let inp_a = dag.add_input(def_a.precision, Shape::number());
            let lut_a = dag.add_lut(inp_a, FunctionTable::UNKWOWN, def_b.precision);
            let weighted_a = dag.add_linear_noise(
                [lut_a],
                LevelledComplexity::ZERO,
                [def_a.norm2.sqrt()],
                Shape::number(),
                "",
            );
            let lut_b = dag.add_lut(weighted_a, FunctionTable::UNKWOWN, def_b.precision);
            let _ = dag.add_linear_noise(
                [lut_b],
                LevelledComplexity::ZERO,
                [def_b.norm2.sqrt()],
                Shape::number(),
                "",
            );
        }
    }
}

/// Adds subgraphs to the dag, to force the generation of fks parameters.
fn build_internal_fks(dag: &mut unparametrized::Dag, internal_partitions: &[InternalPartition]) {
    for def_big in internal_partitions.iter() {
        for def_small in internal_partitions.iter() {
            if def_big > def_small {
                let inp_small = dag.add_input(def_small.precision, Shape::number());
                let lut_small = dag.add_lut(inp_small, FunctionTable::UNKWOWN, def_small.precision);
                let weighted_small = dag.add_linear_noise(
                    [lut_small],
                    LevelledComplexity::ZERO,
                    [def_small.norm2.sqrt()],
                    Shape::number(),
                    "",
                );
                let _ = dag.add_lut(weighted_small, FunctionTable::UNKWOWN, def_small.precision);

                let inp_big = dag.add_input(def_big.precision, Shape::number());
                let lut_big = dag.add_lut(inp_big, FunctionTable::UNKWOWN, def_big.precision);
                let weighted_big = dag.add_linear_noise(
                    [lut_big],
                    LevelledComplexity::ZERO,
                    [def_big.norm2.sqrt()],
                    Shape::number(),
                    "",
                );
                let _ = dag.add_lut(weighted_big, FunctionTable::UNKWOWN, def_big.precision);

                let _ = dag.add_linear_noise(
                    [lut_big, lut_small],
                    LevelledComplexity::ZERO,
                    [0., 0.],
                    Shape::number(),
                    "",
                );
            }
        }
    }
}

fn build_internal_to_external_fks(
    dag: &mut unparametrized::Dag,
    internal_partitions: &[InternalPartition],
    external_partitions: &[ExternalPartition],
) {
    for def_ext in external_partitions {
        for def_int in internal_partitions {
            let inp_int = dag.add_input(def_int.precision, Shape::number());
            let lut_int = dag.add_lut(inp_int, FunctionTable::UNKWOWN, def_int.precision);
            let weighted_int = dag.add_linear_noise(
                [lut_int],
                LevelledComplexity::ZERO,
                [def_int.norm2.sqrt()],
                Shape::number(),
                "",
            );
            let lut_int = dag.add_lut(weighted_int, FunctionTable::UNKWOWN, def_int.precision);
            let weighted_int = dag.add_linear_noise(
                [lut_int],
                LevelledComplexity::ZERO,
                [def_int.norm2.sqrt()],
                Shape::number(),
                "",
            );

            let inp_ext = dag.add_input(0, Shape::number());
            let _ = dag.add_change_partition(inp_ext, Some(def_ext.clone()), None);

            dag.add_composition(weighted_int, inp_ext);
        }
    }
}

/// Adds subgraphs to the dag, to force the generation of external to internal ks parameters.
fn build_external_to_internal_ks(
    dag: &mut unparametrized::Dag,
    internal_partitions: &[InternalPartition],
    external_partitions: &[ExternalPartition],
) {
    for def_ext in external_partitions {
        for def_int in internal_partitions {
            let inp_ext = dag.add_input(def_int.precision, Shape::number());
            let oup_ext = dag.add_change_partition(inp_ext, Some(def_ext.clone()), None);
            let lut_int = dag.add_lut(oup_ext, FunctionTable::UNKWOWN, def_int.precision);
            let _ = dag.add_linear_noise(
                [lut_int],
                LevelledComplexity::ZERO,
                [def_int.norm2.sqrt()],
                Shape::number(),
                "",
            );
        }
    }
}

fn build_virtual_circuit(
    internal_partitions: &[InternalPartition],
    external_partitions: &[ExternalPartition],
) -> unparametrized::Dag {
    let mut dag = unparametrized::Dag::new();
    build_internal_pbs_ks(&mut dag, internal_partitions);
    build_internal_fks(&mut dag, internal_partitions);
    build_internal_to_external_fks(&mut dag, internal_partitions, external_partitions);
    build_external_to_internal_ks(&mut dag, internal_partitions, external_partitions);
    dag
}

pub fn generate_virtual_parameters(
    mut internal_partitions: Vec<InternalPartition>,
    external_partitions: Vec<ExternalPartition>,
    config: Config,
) -> CircuitKeys {
    internal_partitions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    internal_partitions
        .iter()
        .zip(internal_partitions.iter().skip(1))
        .for_each(|(l, r)| {
            assert_ne!(
                l.precision, r.precision,
                "Only one partition by precision can be specified."
            )
        });

    let dag = build_virtual_circuit(
        internal_partitions.as_slice(),
        external_partitions.as_slice(),
    );

    let n_internal_partitions = internal_partitions.len();
    let n_external_partitions = external_partitions.len();

    let precisions = internal_partitions
        .iter()
        .map(|i| i.precision)
        .collect::<Vec<_>>();
    let p_cut =
        PartitionCut::from_precisions_and_external_partitions(&precisions, &external_partitions);

    let search_space = SearchSpace::default_cpu();
    let cache = decomposition::cache(128, ProcessingUnit::Cpu, None, true, 64, 53);
    let parameters = optimize(
        &dag,
        config,
        &search_space,
        &NoSearchSpaceRestriction,
        &cache,
        &Some(p_cut),
        PartitionIndex(0),
    )
    .unwrap()
    .1;

    assert_eq!(
        parameters.macro_params.len(),
        n_internal_partitions + n_external_partitions
    );

    for i in 0..n_internal_partitions {
        for j in 0..n_internal_partitions {
            assert!(
                parameters.micro_params.ks[i][j].is_some(),
                "Internal Ksk[{i},{j}] missing."
            );
            if i > j {
                assert!(
                    parameters.micro_params.fks[i][j].is_some(),
                    "Internal Fksk[{i},{j}] missing."
                );
            }
        }
    }
    for i in 0..n_internal_partitions {
        for j in 0..n_external_partitions {
            assert!(
                parameters.micro_params.fks[i][n_internal_partitions + j].is_some(),
                "Internal -> External Fksk[{i},{}] missing.",
                n_internal_partitions + j
            );
            assert!(
                parameters.micro_params.ks[n_internal_partitions + j][i].is_some(),
                "External -> Internal Ksk[{},{i}] missing.",
                n_internal_partitions + j
            );
        }
    }
    ExpandedCircuitKeys::of(&parameters).compacted()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::computing_cost::cpu::CpuComplexity;
    use crate::optimization::dag::multi_parameters::partitionning::tests::{
        get_tfhers_noise_br, TFHERS_MACRO_PARAMS,
    };

    #[test]
    fn test_generate_generic_parameters_without_externals() {
        let config = Config {
            security_level: 128,
            maximum_acceptable_error_probability: _4_SIGMA,
            key_sharing: true,
            ciphertext_modulus_log: 64,
            fft_precision: 53,
            complexity_model: &CpuComplexity::default(),
        };
        let _a = generate_virtual_parameters(
            vec![
                InternalPartition {
                    precision: 3,
                    norm2: 10.,
                },
                InternalPartition {
                    precision: 4,
                    norm2: 100.,
                },
                InternalPartition {
                    precision: 5,
                    norm2: 100.,
                },
            ],
            vec![],
            config,
        );
    }

    #[test]
    fn test_generate_generic_parameters_with_external() {
        let config = Config {
            security_level: 128,
            maximum_acceptable_error_probability: _4_SIGMA,
            key_sharing: true,
            ciphertext_modulus_log: 64,
            fft_precision: 53,
            complexity_model: &CpuComplexity::default(),
        };
        let variance = get_tfhers_noise_br();
        let _a = generate_virtual_parameters(
            vec![
                InternalPartition {
                    precision: 3,
                    norm2: 10.,
                },
                InternalPartition {
                    precision: 5,
                    norm2: 100.,
                },
            ],
            vec![ExternalPartition {
                name: String::from("tfhers"),
                macro_params: TFHERS_MACRO_PARAMS,
                max_variance: variance,
                variance,
            }],
            config,
        );
    }
}
