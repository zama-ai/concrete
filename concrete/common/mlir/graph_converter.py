"""Module that provides OPGraph conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import concrete.lang as concretelang
import networkx as nx
from mlir.dialects import builtin
from mlir.ir import Context, InsertionPoint, Location, Module

from ..operator_graph import OPGraph
from ..representation.intermediate import Input, IntermediateNode
from .conversion_helpers import value_to_mlir_type
from .node_converter import IntermediateNodeConverter

# pylint: enable=no-name-in-module


class OPGraphConverter(ABC):
    """Converter of OPGraph to MLIR."""

    def convert(self, op_graph: OPGraph) -> str:
        """Convert an operation graph to its corresponding MLIR representation.

        Args:
            op_graph (OPGraph): the operation graph to be converted

        Returns:
            str: textual MLIR representation corresponding to given operation graph
        """

        additional_conversion_info = self._generate_additional_info_dict(op_graph)

        # There are no tensor +*- scalar operations in the compiler
        # But such operations are used commonly so we need to support them
        # So, we implemented some workarounds (pull request #970)
        # Once we have native support, this workaround shall be removed (issue #837)
        # (most changes in #970 shall be reverted)

        # { node1: "%arg0", node2: "%0", node3: "%1" }
        nodes_to_mlir_names: Dict[IntermediateNode, str] = {}

        # { "%arg0": "i5", "%0": "tensor<2x3x!FHE.eint<4>>" }
        mlir_names_to_mlir_types: Dict[str, str] = {}

        # { "%0": ["%c1_i5"] } == for %0 we need to convert %c1_i5 to 1d tensor
        scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]] = {}

        with Context() as ctx, Location.unknown():
            concretelang.register_dialects(ctx)

            module = Module.create()
            with InsertionPoint(module.body):
                parameters = [
                    value_to_mlir_type(ctx, input_node.outputs[0])
                    for input_node in op_graph.get_ordered_inputs()
                ]

                @builtin.FuncOp.from_py_func(*parameters)
                def main(*arg):
                    ir_to_mlir = {}
                    for arg_num, node in op_graph.input_nodes.items():
                        ir_to_mlir[node] = arg[arg_num]

                        mlir_name = f"%arg{arg_num}"
                        nodes_to_mlir_names[node] = mlir_name
                        mlir_names_to_mlir_types[mlir_name] = str(parameters[arg_num])

                    for node in nx.topological_sort(op_graph.graph):
                        if isinstance(node, Input):
                            continue

                        preds = [ir_to_mlir[pred] for pred in op_graph.get_ordered_preds(node)]
                        node_converter = IntermediateNodeConverter(
                            ctx,
                            op_graph,
                            node,
                            preds,
                            nodes_to_mlir_names,
                            mlir_names_to_mlir_types,
                            scalar_to_1d_tensor_conversion_hacks,
                        )
                        ir_to_mlir[node] = node_converter.convert(additional_conversion_info)

                    results = (
                        ir_to_mlir[output_node] for output_node in op_graph.get_ordered_outputs()
                    )
                    return results

        module_lines_after_hacks_are_applied = []
        for line in str(module).split("\n"):
            mlir_name = line.split("=")[0].strip()
            if mlir_name not in scalar_to_1d_tensor_conversion_hacks:
                module_lines_after_hacks_are_applied.append(line)
                continue

            to_be_replaced = scalar_to_1d_tensor_conversion_hacks[mlir_name]
            for arg_name in to_be_replaced:
                new_name = f"%hack_{mlir_name.replace('%', '')}_{arg_name.replace('%', '')}"
                mlir_type = mlir_names_to_mlir_types[arg_name]

                hack_line = (
                    f"    {new_name} = tensor.from_elements {arg_name} : tensor<1x{mlir_type}>"
                )
                module_lines_after_hacks_are_applied.append(hack_line)

                line = line.replace(arg_name, new_name)

            new_arg_types = []

            arg_types = line.split(":")[1].split("->")[0].strip()[1:-1]
            for arg in arg_types.split(", "):
                if arg.startswith("tensor"):
                    new_arg_types.append(arg)
                else:
                    new_arg_types.append(f"tensor<1x{arg}>")

            line = line.replace(arg_types, ", ".join(new_arg_types))

            module_lines_after_hacks_are_applied.append(line)

        return "\n".join(module_lines_after_hacks_are_applied)

    @staticmethod
    @abstractmethod
    def _generate_additional_info_dict(op_graph: OPGraph) -> Dict[str, Any]:
        """Generate additional conversion info dict for the MLIR converter.

        Args:
            op_graph (OPGraph): the operation graph from which the additional info will be generated

        Returns:
            Dict[str, Any]: dict of additional conversion info
        """
