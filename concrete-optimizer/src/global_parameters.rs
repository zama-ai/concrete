use std::collections::HashSet;

use crate::dag::operator::{Operator, OperatorIndex};
use crate::dag::parameter_indexed::{
    InputParameterIndexed, LutParametersIndexed, OperatorParameterIndexed,
};
use crate::dag::unparametrized::UnparameterizedOperator;
use crate::dag::{parameter_indexed, range_parametrized, unparametrized};
use crate::parameters::{
    BrDecompositionParameterRanges, BrDecompositionParameters, GlweParameterRanges, GlweParameters,
    KsDecompositionParameterRanges, KsDecompositionParameters,
};

#[derive(Clone)]
pub(crate) struct ParameterToOperation {
    pub glwe: Vec<Vec<OperatorIndex>>,
    pub br_decomposition: Vec<Vec<OperatorIndex>>,
    pub ks_decomposition: Vec<Vec<OperatorIndex>>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParameterCount {
    pub glwe: usize,
    pub br_decomposition: usize,
    pub ks_decomposition: usize,
}

#[derive(Clone)]
pub struct ParameterRanges {
    pub glwe: Vec<GlweParameterRanges>,
    pub br_decomposition: Vec<BrDecompositionParameterRanges>, // 0 => lpetit , 1 => l plus grand
    pub ks_decomposition: Vec<KsDecompositionParameterRanges>,
}

pub struct ParameterValues {
    pub glwe: Vec<GlweParameters>,
    pub br_decomposition: Vec<BrDecompositionParameters>,
    pub ks_decomposition: Vec<KsDecompositionParameters>,
}

#[derive(Clone, Copy)]
pub struct ParameterDomains {
    // move next comment to pareto ranges definition
    // TODO: verify if pareto optimal parameters depends on precisions
    pub glwe_pbs_constrained: GlweParameterRanges,
    pub free_glwe: GlweParameterRanges,
    pub br_decomposition: BrDecompositionParameterRanges,
    pub ks_decomposition: KsDecompositionParameterRanges,
}

pub const DEFAUT_DOMAINS: ParameterDomains = ParameterDomains {
    glwe_pbs_constrained: GlweParameterRanges {
        log2_polynomial_size: Range { start: 10, end: 15 },
        glwe_dimension: Range { start: 1, end: 7 },
    },
    free_glwe: GlweParameterRanges {
        log2_polynomial_size: Range { start: 0, end: 1 },
        glwe_dimension: Range {
            start: 512,
            end: 1025,
        },
    },
    br_decomposition: BrDecompositionParameterRanges {
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
pub fn minimal_unify(_g: unparametrized::OperationDag) -> parameter_indexed::OperationDag {
    todo!()
}

fn convert_maximal(op: UnparameterizedOperator) -> OperatorParameterIndexed {
    let external_glwe_index = 0;
    let internal_lwe_index = 1;
    let br_decomposition_index = 0;
    let ks_decomposition_index = 0;
    match op {
        Operator::Input {
            out_precision,
            out_shape,
            ..
        } => Operator::Input {
            out_precision,
            out_shape,
            extra_data: InputParameterIndexed {
                lwe_dimension_index: external_glwe_index,
            },
        },
        Operator::Lut {
            input,
            table,
            out_precision,
            ..
        } => Operator::Lut {
            input,
            table,
            out_precision,
            extra_data: LutParametersIndexed {
                input_lwe_dimension_index: external_glwe_index,
                ks_decomposition_parameter_index: ks_decomposition_index,
                internal_lwe_dimension_index: internal_lwe_index,
                br_decomposition_parameter_index: br_decomposition_index,
                output_glwe_params_index: external_glwe_index,
            },
        },
        Operator::Dot {
            inputs, weights, ..
        } => Operator::Dot {
            inputs,
            weights,
            extra_data: (),
        },
        Operator::LevelledOp {
            inputs,
            complexity,
            manp,
            out_shape,
            comment,
            ..
        } => Operator::LevelledOp {
            inputs,
            complexity,
            manp,
            comment,
            out_shape,
            extra_data: (),
        },
    }
}

#[must_use]
pub fn maximal_unify(g: unparametrized::OperationDag) -> parameter_indexed::OperationDag {
    let operators: Vec<_> = g.operators.into_iter().map(convert_maximal).collect();

    let parameters = ParameterCount {
        glwe: 2,
        br_decomposition: 1,
        ks_decomposition: 1,
    };

    let mut reverse_map = ParameterToOperation {
        glwe: vec![vec![], vec![]],
        br_decomposition: vec![vec![]],
        ks_decomposition: vec![vec![]],
    };

    for (i, op) in operators.iter().enumerate() {
        let index = OperatorIndex { i };
        match op {
            Operator::Input { .. } => {
                reverse_map.glwe[0].push(index);
            }
            Operator::Lut { .. } => {
                reverse_map.glwe[0].push(index);
                reverse_map.glwe[1].push(index);
                reverse_map.br_decomposition[0].push(index);
                reverse_map.ks_decomposition[0].push(index);
            }
            Operator::Dot { .. } | Operator::LevelledOp { .. } => {
                reverse_map.glwe[0].push(index);
                reverse_map.glwe[1].push(index);
            }
        }
    }

    parameter_indexed::OperationDag {
        operators,
        parameters_count: parameters,
        reverse_map,
    }
}

#[must_use]
pub fn domains_to_ranges(
    parameter_indexed::OperationDag {
        operators,
        parameters_count,
        reverse_map,
    }: parameter_indexed::OperationDag,
    domains: ParameterDomains,
) -> range_parametrized::OperationDag {
    let mut constrained_glwe_parameter_indexes = HashSet::new();
    for op in &operators {
        if let Operator::Lut { extra_data, .. } = op {
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
        br_decomposition: vec![
            domains.br_decomposition;
            parameters_count.br_decomposition as usize
        ],
        ks_decomposition: vec![domains.ks_decomposition; parameters_count.ks_decomposition],
    };

    range_parametrized::OperationDag {
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
    use crate::dag::operator::{FunctionTable, LevelledComplexity, Shape};

    #[test]
    fn test_maximal_unify() {
        let mut graph = unparametrized::OperationDag::new();

        let input1 = graph.add_input(1, Shape::number());

        let input2 = graph.add_input(2, Shape::number());

        let cpx_add = LevelledComplexity::ADDITION;
        let sum1 = graph.add_levelled_op([input1, input2], cpx_add, 1.0, Shape::number(), "sum");

        let lut1 = graph.add_lut(sum1, FunctionTable::UNKWOWN, 2);

        let concat = graph.add_levelled_op([input1, lut1], cpx_add, 1.0, Shape::number(), "concat");

        let dot = graph.add_dot([concat], [1, 2]);

        let lut2 = graph.add_lut(dot, FunctionTable::UNKWOWN, 2);

        let graph_params = maximal_unify(graph);

        assert_eq!(
            graph_params.parameters_count,
            ParameterCount {
                glwe: 2,
                br_decomposition: 1,
                ks_decomposition: 1,
            }
        );

        assert_eq!(
            graph_params.reverse_map.glwe,
            vec![
                vec![input1, input2, sum1, lut1, concat, dot, lut2],
                vec![sum1, lut1, concat, dot, lut2]
            ]
        );

        assert_eq!(
            graph_params.reverse_map.br_decomposition,
            vec![vec![lut1, lut2]]
        );

        assert_eq!(
            graph_params.reverse_map.ks_decomposition,
            vec![vec![lut1, lut2]]
        );
        // collectes l'ensemble des parametres
        // unify structurellement les parametres identiques
        // => counts
        // =>
        // let parametrized_expr = { global, dag + indexation}
    }

    #[test]
    fn test_simple_lwe() {
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let _input2 = graph.add_input(2, Shape::number());

        let graph_params = maximal_unify(graph);

        let range_parametrized::OperationDag {
            operators,
            parameter_ranges,
            reverse_map: _,
        } = domains_to_ranges(graph_params, DEFAUT_DOMAINS);

        let input_1_lwe_params = match &operators[input1.i] {
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
        let mut graph = unparametrized::OperationDag::new();
        let input1 = graph.add_input(1, Shape::number());
        let input2 = graph.add_input(2, Shape::number());

        let cpx_add = LevelledComplexity::ADDITION;
        let concat =
            graph.add_levelled_op([input1, input2], cpx_add, 1.0, Shape::vector(2), "concat");

        let lut1 = graph.add_lut(concat, FunctionTable::UNKWOWN, 2);

        let graph_params = maximal_unify(graph);

        let range_parametrized::OperationDag {
            operators,
            parameter_ranges,
            reverse_map: _,
        } = domains_to_ranges(graph_params, DEFAUT_DOMAINS);

        let input_1_lwe_params = match &operators[input1.i] {
            Operator::Input { extra_data, .. } => extra_data.lwe_dimension_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.glwe_pbs_constrained,
            parameter_ranges.glwe[input_1_lwe_params]
        );

        let lut1_out_glwe_params = match &operators[lut1.i] {
            Operator::Lut { extra_data, .. } => extra_data.output_glwe_params_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.glwe_pbs_constrained,
            parameter_ranges.glwe[lut1_out_glwe_params]
        );

        let lut1_internal_glwe_params = match &operators[lut1.i] {
            Operator::Lut { extra_data, .. } => extra_data.internal_lwe_dimension_index,
            _ => unreachable!(),
        };
        assert_eq!(
            DEFAUT_DOMAINS.free_glwe,
            parameter_ranges.glwe[lut1_internal_glwe_params]
        );
    }
}
