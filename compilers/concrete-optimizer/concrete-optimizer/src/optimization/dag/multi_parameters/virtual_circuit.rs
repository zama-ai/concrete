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
    partition_cut::PartitionCut,
    PartitionIndex,
};

const _4_SIGMA: f64 = 0.000_063_342_483_999_973;

#[derive(Debug, Clone, PartialEq)]
pub struct PartitionDefinition {
    pub precision: Precision,
    pub norm2: f64,
}

impl PartialOrd for PartitionDefinition {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.precision.cmp(&other.precision) {
            std::cmp::Ordering::Equal => self.norm2.partial_cmp(&other.norm2),
            ordering => Some(ordering),
        }
    }
}

fn generate_virtual_circuit(
    partitions: &[PartitionDefinition],
    generate_fks: bool,
) -> unparametrized::Dag {
    let mut dag = unparametrized::Dag::new();

    for def_a in partitions.iter() {
        for def_b in partitions.iter() {
            if def_a == def_b {
                continue;
            }
            let inp_a = dag.add_input(def_a.precision, Shape::number());
            let lut_a = dag.add_lut(inp_a, FunctionTable::UNKWOWN, def_a.precision);
            let _weighted_a = dag.add_linear_noise(
                [lut_a],
                LevelledComplexity::ZERO,
                [def_a.norm2.sqrt()],
                Shape::number(),
                "",
            );

            let inp_b = dag.add_input(def_b.precision, Shape::number());
            let lut_b = dag.add_lut(inp_b, FunctionTable::UNKWOWN, def_b.precision);
            let weighted_b = dag.add_linear_noise(
                [lut_b],
                LevelledComplexity::ZERO,
                [def_b.norm2.sqrt()],
                Shape::number(),
                "",
            );

            dag.add_composition(weighted_b, inp_a);

            if generate_fks && def_a > def_b {
                let inp_a = dag.add_input(def_a.precision, Shape::number());
                let lut_a = dag.add_lut(inp_a, FunctionTable::UNKWOWN, def_a.precision);
                let _weighted_a = dag.add_linear_noise(
                    [lut_a],
                    LevelledComplexity::ZERO,
                    [def_a.norm2.sqrt()],
                    Shape::number(),
                    "",
                );

                let inp_b = dag.add_input(def_b.precision, Shape::number());
                let lut_b = dag.add_lut(inp_b, FunctionTable::UNKWOWN, def_b.precision);
                let _weighted_b = dag.add_linear_noise(
                    [lut_b],
                    LevelledComplexity::ZERO,
                    [def_b.norm2.sqrt()],
                    Shape::number(),
                    "",
                );

                let _ = dag.add_linear_noise(
                    [lut_a, lut_b],
                    LevelledComplexity::ZERO,
                    [0., 0.],
                    Shape::number(),
                    "",
                );
            }
        }
    }
    dag
}

pub fn generate_virtual_parameters(
    partitions: Vec<PartitionDefinition>,
    generate_fks: bool,
    config: Config,
) -> CircuitKeys {
    let dag = generate_virtual_circuit(partitions.as_slice(), generate_fks);

    let precisions: Vec<_> = partitions.iter().map(|def| def.precision).collect();
    let n_partitions = precisions.len();
    let p_cut = PartitionCut::maximal_partitionning(&dag);
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
    .map_or(None, |v| Some(v.1))
    .unwrap();

    for i in 0..n_partitions {
        for j in 0..n_partitions {
            assert!(
                parameters.micro_params.ks[i][j].is_some(),
                "Ksk[{i},{j}] missing."
            );
            if i > j {
                assert!(
                    parameters.micro_params.fks[i][j].is_some(),
                    "Fksk[{i},{j}] missing."
                );
            }
        }
    }
    ExpandedCircuitKeys::of(&parameters).compacted()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::computing_cost::cpu::CpuComplexity;

    #[test]
    fn test_generate_generic_parameters() {
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
                PartitionDefinition {
                    precision: 3,
                    norm2: 1.,
                },
                PartitionDefinition {
                    precision: 3,
                    norm2: 100.,
                },
                PartitionDefinition {
                    precision: 3,
                    norm2: 1000.,
                },
            ],
            true,
            config,
        );
    }
}
