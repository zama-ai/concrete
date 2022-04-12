use std::collections::HashSet;

use crate::graph::operator::{Operator, OperatorIndex};
use crate::graph::parameter_indexed::{AtomicPatternParametersIndexed, InputParameterIndexed};
use crate::graph::{parameter_indexed, range_parametrized, unparametrized};
use crate::parameters::{
    GlweParameterRanges, GlweParameters, KsDecompositionParameterRanges, KsDecompositionParameters,
    PbsDecompositionParameterRanges, PbsDecompositionParameters,
};

#[derive(Clone)]
pub(crate) struct ParameterToOperation {
    pub glwe: Vec<Vec<OperatorIndex>>,
    pub pbs_decomposition: Vec<Vec<OperatorIndex>>,
    pub ks_decomposition: Vec<Vec<OperatorIndex>>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ParameterCount {
    pub glwe: usize,
    pub pbs_decomposition: usize,
    pub ks_decomposition: usize,
}

#[derive(Clone)]
pub struct ParameterRanges {
    pub glwe: Vec<GlweParameterRanges>,
    pub pbs_decomposition: Vec<PbsDecompositionParameterRanges>, // 0 => lpetit , 1 => l plus grand
    pub ks_decomposition: Vec<KsDecompositionParameterRanges>,
}

pub struct ParameterValues {
    pub glwe: Vec<GlweParameters>,
    pub pbs_decomposition: Vec<PbsDecompositionParameters>,
    pub ks_decomposition: Vec<KsDecompositionParameters>,
}

#[derive(Clone, Copy)]
pub struct ParameterDomains {
    // move next comment to pareto ranges definition
    // TODO: verify if pareto optimal parameters depends on precisions
    pub glwe_pbs_constrained: GlweParameterRanges,
    pub free_glwe: GlweParameterRanges,
    pub pbs_decomposition: PbsDecompositionParameterRanges,
    pub ks_decomposition: KsDecompositionParameterRanges,
}

pub const DEFAUT_DOMAINS: ParameterDomains = ParameterDomains {
    glwe_pbs_constrained: GlweParameterRanges {
        log2_polynomial_size: Range { start: 8, end: 15 },
        glwe_dimension: Range { start: 1, end: 2 },
    },
    free_glwe: GlweParameterRanges {
        log2_polynomial_size: Range { start: 0, end: 1 },
        glwe_dimension: Range {
            start: 512,
            end: 1025,
        },
    },
    pbs_decomposition: PbsDecompositionParameterRanges {
        log2_base: Range { start: 1, end: 65 },
        level: Range { start: 1, end: 65 },
    },
    ks_decomposition: KsDecompositionParameterRanges {
        log2_base: Range { start: 1, end: 65 },
        level: Range { start: 1, end: 65 },
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Range {
    pub start: u64,
    pub end: u64,
}

impl IntoIterator for &Range {
    type Item = u64;

    type IntoIter = std::ops::Range<u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.start..self.end
    }
}

impl Range {
    pub fn as_vec(self) -> Vec<u64> {
        self.into_iter().collect()
    }
}

#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub fn minimal_unify(_g: unparametrized::AtomicPatternDag) -> parameter_indexed::AtomicPatternDag {
    todo!()
}

fn convert_maximal(
    op: Operator<(), ()>,
) -> Operator<InputParameterIndexed, AtomicPatternParametersIndexed> {
    let external_glwe_index = 0;
    let internal_lwe_index = 1;
    let pbs_decomposition_index = 0;
    let ks_decomposition_index = 0;
    match op {
        Operator::Input { out_precision, .. } => Operator::Input {
            out_precision,
            extra_data: InputParameterIndexed {
                lwe_dimension_index: external_glwe_index,
            },
        },
        Operator::AtomicPattern {
            in_precision,
            out_precision,
            multisum_inputs,
            ..
        } => Operator::AtomicPattern {
            in_precision,
            out_precision,
            multisum_inputs,
            extra_data: AtomicPatternParametersIndexed {
                input_lwe_dimensionlwe_dimension_index: external_glwe_index,
                ks_decomposition_parameter_index: ks_decomposition_index,
                internal_lwe_dimension_index: internal_lwe_index,
                pbs_decomposition_parameter_index: pbs_decomposition_index,
                output_glwe_params_index: external_glwe_index,
            },
        },
    }
}

#[must_use]
pub fn maximal_unify(g: unparametrized::AtomicPatternDag) -> parameter_indexed::AtomicPatternDag {
    let operators: Vec<_> = g.operators.into_iter().map(convert_maximal).collect();

    let parameters = ParameterCount {
        glwe: 2,
        pbs_decomposition: 1,
        ks_decomposition: 1,
    };

    let mut reverse_map = ParameterToOperation {
        glwe: vec![vec![], vec![]],
        pbs_decomposition: vec![vec![]],
        ks_decomposition: vec![vec![]],
    };

    for (i, op) in operators.iter().enumerate() {
        match op {
            Operator::Input { .. } => {
                reverse_map.glwe[0].push(OperatorIndex(i));
            }
            Operator::AtomicPattern { .. } => {
                reverse_map.glwe[0].push(OperatorIndex(i));
                reverse_map.glwe[1].push(OperatorIndex(i));
                reverse_map.pbs_decomposition[0].push(OperatorIndex(i));
                reverse_map.ks_decomposition[0].push(OperatorIndex(i));
            }
        }
    }

    parameter_indexed::AtomicPatternDag {
        operators,
        parameters_count: parameters,
        reverse_map,
    }
}

#[must_use]
pub fn domains_to_ranges(
    parameter_indexed::AtomicPatternDag {
        operators,
        parameters_count,
        reverse_map,
    }: parameter_indexed::AtomicPatternDag,
    domains: ParameterDomains,
) -> range_parametrized::AtomicPatternDag {
    let mut constrained_glwe_parameter_indexes = HashSet::new();
    for op in &operators {
        if let Operator::AtomicPattern { extra_data, .. } = op {
            let _ = constrained_glwe_parameter_indexes.insert(extra_data.output_glwe_params_index);
        }
    }

    let mut glwe = vec![];

    for i in 0..parameters_count.glwe {
        if constrained_glwe_parameter_indexes.contains(&i) {
            glwe.push(domains.glwe_pbs_constrained);
        } else {
            glwe.push(domains.free_glwe);
        }
    }

    let parameter_ranges = ParameterRanges {
        glwe,
        pbs_decomposition: vec![
            domains.pbs_decomposition;
            parameters_count.pbs_decomposition as usize
        ],
        ks_decomposition: vec![domains.ks_decomposition; parameters_count.ks_decomposition],
    };

    range_parametrized::AtomicPatternDag {
        operators,
        parameter_ranges,
        reverse_map,
    }
}

// fn fill_ranges(g: parameter_indexed::AtomicPatternDag) ->  parameter_ranged::AtomicPatternDag {
//     //check unconstrained GlweDim -> set range_poly_size=[1, 2[
//     todo!()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::Weight;

    #[test]
    fn test_maximal_unify() {
        let mut graph = unparametrized::AtomicPatternDag::new();

        let input1 = graph.add_input(1);

        let input2 = graph.add_input(2);

        let atomic_pattern1 =
            graph.add_atomic_pattern(3, 3, vec![(Weight(1), input1), (Weight(2), input2)]);

        let _atomic_pattern2 = graph.add_atomic_pattern(
            4,
            4,
            vec![(Weight(1), atomic_pattern1), (Weight(2), input2)],
        );

        let graph_params = maximal_unify(graph);

        assert_eq!(
            graph_params.parameters_count,
            ParameterCount {
                glwe: 2,
                pbs_decomposition: 1,
                ks_decomposition: 1,
            }
        );

        assert_eq!(
            graph_params.reverse_map.glwe,
            vec![
                vec![
                    OperatorIndex(0),
                    OperatorIndex(1),
                    OperatorIndex(2),
                    OperatorIndex(3)
                ],
                vec![OperatorIndex(2), OperatorIndex(3)]
            ]
        );

        assert_eq!(
            graph_params.reverse_map.pbs_decomposition,
            vec![vec![OperatorIndex(2), OperatorIndex(3)]]
        );

        assert_eq!(
            graph_params.reverse_map.ks_decomposition,
            vec![vec![OperatorIndex(2), OperatorIndex(3)]]
        );
        // collectes l'ensemble des parametres
        // unify structurellement les parametres identiques
        // => counts
        // =>
        // let parametrized_expr = { global, dag + indexation}
    }

    #[test]
    fn test_simple_lwe() {
        let mut graph = unparametrized::AtomicPatternDag::new();
        let input1 = graph.add_input(1);
        let _input2 = graph.add_input(2);

        let graph_params = maximal_unify(graph);

        let range_parametrized::AtomicPatternDag {
            operators,
            parameter_ranges,
            reverse_map: _,
        } = domains_to_ranges(graph_params, DEFAUT_DOMAINS);

        let input_1_lwe_params = match &operators[input1.0] {
            Operator::Input { extra_data, .. } => extra_data.lwe_dimension_index,
            _ => unreachable!(),
        };

        dbg!(&parameter_ranges.glwe);

        assert_eq!(
            DEFAUT_DOMAINS.free_glwe,
            parameter_ranges.glwe[input_1_lwe_params]
        );
    }

    #[test]
    fn test_simple_lwe2() {
        let mut graph = unparametrized::AtomicPatternDag::new();
        let input1 = graph.add_input(1);
        let input2 = graph.add_input(2);

        let atomic_pattern1 =
            graph.add_atomic_pattern(3, 3, vec![(Weight(1), input1), (Weight(2), input2)]);

        let graph_params = maximal_unify(graph);

        let range_parametrized::AtomicPatternDag {
            operators,
            parameter_ranges,
            reverse_map: _,
        } = domains_to_ranges(graph_params, DEFAUT_DOMAINS);

        let input_1_lwe_params = match &operators[input1.0] {
            Operator::Input { extra_data, .. } => extra_data.lwe_dimension_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.glwe_pbs_constrained,
            parameter_ranges.glwe[input_1_lwe_params]
        );

        let atomic_pattern1_out_glwe_params = match &operators[atomic_pattern1.0] {
            Operator::AtomicPattern { extra_data, .. } => extra_data.output_glwe_params_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.glwe_pbs_constrained,
            parameter_ranges.glwe[atomic_pattern1_out_glwe_params]
        );

        let atomic_pattern1_internal_glwe_params = match &operators[atomic_pattern1.0] {
            Operator::AtomicPattern { extra_data, .. } => extra_data.internal_lwe_dimension_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.free_glwe,
            parameter_ranges.glwe[atomic_pattern1_internal_glwe_params]
        );
    }
}
