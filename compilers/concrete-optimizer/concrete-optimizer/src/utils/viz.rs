use crate::dag::operator::Operator;

/// A trait allowing to visualize objects as graphviz/dot graphs.
///
/// Useful to use along with the [`viz`] and [`vizp`] macros to debug objects.
pub trait Viz {
    /// This method must return a string on the `dot` format, containing the description of the
    /// object.
    fn viz_node(&self) -> String;

    /// This method can be re-implemented if the object must be referenced by other objects `Viz`
    /// implementation (to create edges mostly).
    fn viz_label(&self) -> String {
        String::new()
    }

    /// Returns a string containing a valid dot representation of the object.
    fn viz_string(&self) -> String {
        format!(
            "
strict digraph G {{
fontname=\"Helvetica,Arial,sans-serif\"
node [fontname=\"Helvetica,Arial,sans-serif\" gradientangle=270]
edge [fontname=\"Helvetica,Arial,sans-serif\"]
rankdir=TB;
node [shape=record style=filled];
{}
}}",
            self.viz_node()
        )
    }
}

impl Viz for crate::dag::unparametrized::Dag {
    fn viz_node(&self) -> String {
        let mut graph = vec![];
        self.get_circuits_iter()
            .for_each(|circuit| graph.push(circuit.viz_node()));
        self.composition
            .clone()
            .into_iter()
            .flat_map(|(to, froms)| froms.into_iter().map(move |f| (to, f)))
            .for_each(|(to, from)| {
                graph.push(format!("{from} -> {to} [color=red, style=dashed]"));
            });
        graph.join("\n")
    }
}

impl<'dag> Viz for crate::dag::unparametrized::DagCircuit<'dag> {
    fn viz_node(&self) -> String {
        let mut graph: Vec<String> = vec![];
        let circuit = &self.circuit;
        self.get_operators_iter().for_each(|node| {
            graph.push(node.viz_node());
            node.get_inputs_iter().for_each(|inp_node| {
                let inp_label = inp_node.viz_label();
                let oup_label = node.viz_label();
                graph.push(format!("{inp_label} -> {oup_label} [weight=10];"));
            });
        });
        format!(
            "
subgraph cluster_circuit_{circuit} {{
label=\"Circuit {circuit}\"
style=\"rounded\"
{}
}}
",
            graph.join("\n")
        )
    }
}

impl<'dag> Viz for crate::dag::unparametrized::DagOperator<'dag> {
    fn viz_node(&self) -> String {
        let input_string = self
            .operator
            .get_inputs_iter()
            .map(|id| id.0)
            .map(|n| format!("%{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let index = self.id;
        let color = if self.is_input() || self.is_output() {
            "lightseagreen"
        } else {
            "lightgreen"
        };
        match self.operator {
            Operator::Input { out_precision, .. } => {
                format!("{index} [label =\"{{%{index} = Input({input_string}) |{{out_precision:|{out_precision:?}}}}}\" fillcolor={color}];")
            }
            Operator::Lut { out_precision, .. } => {
                format!("{index} [label = \"{{%{index} = Lut({input_string}) |{{out_precision:|{out_precision:?}}}}}\" fillcolor={color}];")
            }
            Operator::Dot { .. } => {
                format!("{index} [label = \"{{%{index} = Dot({input_string})}}\" fillcolor={color}];")
            }
            Operator::LevelledOp { manp, .. } => {
                format!("{index} [label = \"{{%{index} = LevelledOp({input_string}) |{{manp:|{manp:?}}}}}\" fillcolor={color}];")
            }
            Operator::UnsafeCast { out_precision, .. } => format!(
                "{index} [label = \"{{%{index} = UnsafeCast({input_string}) |{{out_precision:|{out_precision:?}}}}}\" fillcolor={color}];"
            ),
            Operator::Round { out_precision, .. } => {
                format!("{index} [label = \"{{%{index} = Round({input_string}) |{{out_precision:|{out_precision:?}}}}}\" fillcolor={color}];",)
            }
        }
    }

    fn viz_label(&self) -> String {
        let index = self.id;
        format!("{index}")
    }
}

impl Viz for crate::optimization::dag::multi_parameters::analyze::PartitionedDag {
    fn viz_node(&self) -> String {
        let mut output = self.dag.viz_node();
        self.partitions
            .instrs_partition
            .iter()
            .enumerate()
            .for_each(|(i, part)| {
                let partition = part.instruction_partition;
                let circuit = &self.dag.circuit_tags[i];
                // let color = partition.0 + 1;
                output.push_str(&format!("subgraph cluster_circuit_{circuit} {{\n"));
                output.push_str(&format!("partition_{i} [label =\"{partition}\"];\n"));
                output.push_str(&format!(
                    "partition_{i} -> {i} [arrowhead=none, color=gray80, weight=99];\n"
                ));
                output.push_str("}\n");
            });
        output
    }
}

impl Viz for crate::optimization::dag::multi_parameters::analyze::VariancedDag {
    fn viz_node(&self) -> String {
        let mut output = self.dag.viz_node();
        self.partitions
            .instrs_partition
            .iter()
            .zip(self.variances.vars.iter())
            .enumerate()
            .for_each(|(i, (part, var))| {
                let partition = part.instruction_partition;
                let circuit = &self.dag.circuit_tags[i];
                let variances = var
                    .vars
                    .iter()
                    .enumerate()
                    .map(|(i, var)| format!("{{{i}|{var}}}"))
                    .collect::<Vec<_>>()
                    .join("|");
                let label = format!(
                    "<{{<b>Partition</b> | {partition} | <b>Variances</b> | {variances} }}>"
                );
                output.push_str(&format!("subgraph cluster_circuit_{circuit} {{\n"));
                output.push_str(&format!(
                    "info_{i} [label ={label} color=gray80 fillcolor=gray90];\n"
                ));
                output.push_str(&format!(
                    "{i} -> info_{i} [arrowhead=none, color=gray90, weight=99];\n"
                ));
                output.push_str("}\n");
            });
        output
    }
}

macro_rules! _viz {
    ($path: expr, $object:expr) => {{
        let mut path = std::env::temp_dir();
        path.push(AsRef::<std::path::Path>::as_ref($path));
        let _ = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!(
                "echo '{}' | dot -Tsvg > {}",
                $crate::utils::viz::Viz::viz_string($object),
                path.to_str().unwrap()
            ))
            .output()
            .expect("Failed to execute dot. Do you have graphviz installed ?");
    }};
}

/// Dumps the visualization of an object to a given svg file.
#[allow(unused)]
macro_rules! viz {
    ($path: expr, $object:expr) => {
        $crate::utils::viz::_viz!($path, $object);
        println!(
            "Viz of {}:{} visible at {}/{}",
            file!(),
            line!(),
            std::env::temp_dir().display(),
            $path
        );
    };
    ($object:expr) => {
        let name = format!("concrete_optimizer_dbg_{}.svg", rand::random::<u64>());
        $crate::utils::viz::viz!(&name, $object);
    };
}

/// Dumps the visualization of an object to a given svg file and panics.
#[allow(unused)]
macro_rules! vizp {
    ($path: expr, $object:expr) => {{
        $crate::utils::viz::_viz!($path, $object);
        panic!(
            "Viz of {}:{} visible at {}/{}",
            file!(),
            line!(),
            std::env::temp_dir().display(),
            $path
        );
    }};
    ($object:expr) => {
        let name = format!("concrete_optimizer_dbg_{}.svg", rand::random::<u64>());
        $crate::utils::viz::vizp!(&name, $object);
    };
}

#[allow(unused)]
pub(crate) use _viz;
#[allow(unused)]
pub(crate) use viz;
#[allow(unused)]
pub(crate) use vizp;
