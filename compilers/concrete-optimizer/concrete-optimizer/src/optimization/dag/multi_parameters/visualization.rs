use std::iter::once;
use std::path::PathBuf;
use std::process::Command;

use super::analyze::AnalyzedDag;
use super::partitions::Transition;
use crate::dag::operator::Operator;

const COLORSCHEME: &str = "set19";

#[allow(unused)]
/// Saves a graphviz representation of the analyzed dag.
///
/// The image is stored in your temp folder as `concrete_optimizer_dbg.png`.
pub fn save_dag_dot(dag: &AnalyzedDag) {
    let path = write_dot_svg(&analyzed_dag_to_dot_string(dag), None);
    println!("Analyzed dag visible at path {}", path.to_str().unwrap());
}

#[allow(unused)]
/// Dump the dag dot code and panic.
///
/// For debug purpose essentially.
pub fn dump_dag_dot(dag: &AnalyzedDag) {
    println!("{}", analyzed_dag_to_dot_string(dag));
    panic!();
}

fn write_dot_svg(dot: &str, maybe_path: Option<PathBuf>) -> PathBuf {
    let mut path = maybe_path.unwrap_or_else(std::env::temp_dir);
    path.push("concrete_optimizer_dbg.png");
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "echo '{}' | dot -Tpng > {}",
            dot,
            path.to_str().unwrap()
        ))
        .output()
        .expect("Failed to execute dot. Do you have graphviz installed ?");
    path
}

fn extract_node_inputs(node: &Operator) -> Vec<usize> {
    node.get_inputs_iter().map(|id| id.i).collect()
}

fn extract_node_label(node: &Operator, index: usize) -> String {
    let input_string = extract_node_inputs(node)
        .iter()
        .map(|n| format!("%{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    match node {
        Operator::Input { out_precision, .. } => {
            format!("{{%{index} = Input({input_string}) |{{out_precision:|{out_precision:?}}}}}",)
        }
        Operator::Lut { out_precision, .. } => {
            format!("{{%{index} = Lut({input_string}) |{{out_precision:|{out_precision:?}}}}}",)
        }
        Operator::Dot { .. } => {
            format!("{{%{index} = Dot({input_string})}}")
        }
        Operator::LevelledOp { manp, .. } => {
            format!("{{%{index} = LevelledOp({input_string}) |{{manp:|{manp:?}}}}}",)
        }
        Operator::UnsafeCast { out_precision, .. } => format!(
            "{{%{index} = UnsafeCast({input_string}) |{{out_precision:|{out_precision:?}}}}}",
        ),
        Operator::Round { out_precision, .. } => {
            format!("{{%{index} = Round({input_string}) |{{out_precision:|{out_precision:?}}}}}",)
        }
    }
}

fn analyzed_dag_to_dot_string(dag: &AnalyzedDag) -> String {
    let partitions: Vec<String> = dag
        .p_cut
        .p_cut
        .iter()
        .map(|(p, _)| format!("{p}"))
        .chain(once(String::new()))
        .enumerate()
        .map(|(i, pci)| {
            format!(
                "partition_{i} [label=\"{{ Partition {i} | {{ p_cut: | {pci} }} }}\" fillcolor={}]",
                i + 1
            )
        })
        .collect();

    let mut graph: Vec<String> = vec![];
    let iterator = dag
        .operators
        .iter()
        .zip(dag.instrs_partition.iter())
        .enumerate();

    for (i, (node, partition)) in iterator {
        if let Operator::Lut { input, .. } = node {
            let input_partition_color = dag.instrs_partition[input.i].instruction_partition + 1;
            let lut_partition_color = partition.instruction_partition + 1;
            let input_index = input.i;
            let label = extract_node_label(node, i);
            let node = format!("
                    {input_index} -> ks_{i} [color=\"/{COLORSCHEME}/{input_partition_color}\"];
                    subgraph cluster_{i}{{
                        label_{i} [label=\"{label}\"];
                        ks_{i} [label=\"KS\" fillcolor=\"/{COLORSCHEME}/{input_partition_color}:/{COLORSCHEME}/{lut_partition_color}\"];
                        {i} [label=\"MS\\+BR\" fillcolor={lut_partition_color}];
                        ks_{i} -> {i} [color=\"/{COLORSCHEME}/{lut_partition_color}\"];
                    }}
                ");
            graph.push(node);
            for fks_i in &partition.alternative_output_representation {
                let output_partition_color = fks_i + 1;
                graph.push(format!("
                        fks_{i}_{fks_i} [label=\"FKS\", fillcolor=\"/{COLORSCHEME}/{lut_partition_color}:/{COLORSCHEME}/{output_partition_color}\"];
                        {i} -> fks_{i}_{fks_i} [color=\"/{COLORSCHEME}/{lut_partition_color}\"];
                    "));
            }
        } else {
            let partition_index = partition.instruction_partition;
            let partition_color = partition_index + 1;
            let label = extract_node_label(node, i);
            graph.push(format!(
                "{i} [label=\"{label}\" fillcolor={partition_color}];",
            ));

            for (j, input_index) in extract_node_inputs(node).into_iter().enumerate() {
                match partition.inputs_transition.get(j) {
                    Some(Some(Transition::Additional { .. })) => {
                        graph.push(format!("fks_{input_index}_{partition_index} -> {i} [color=\"/{COLORSCHEME}/{partition_color}\"]"));
                    }
                    _ => {
                        graph.push(format!(
                            "{input_index} -> {i} [color=\"/{COLORSCHEME}/{partition_color}\"]",
                        ));
                    }
                };
            }
        }
    }

    format!(
        "
digraph G {{
fontname=\"Helvetica,Arial,sans-serif\"
node [fontname=\"Helvetica,Arial,sans-serif\" gradientangle=270]
edge [fontname=\"Helvetica,Arial,sans-serif\"]
rankdir=TB;
node [shape=record colorscheme={COLORSCHEME} style=filled];
subgraph cluster_0 {{
label=\"Partitions\"
{}
}}
{}
}}",
        partitions.join("\n"),
        graph.join("\n"),
    )
}
