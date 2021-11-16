"""Module that provides OPGraph conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from abc import ABC, abstractmethod
from typing import Any, Dict

import networkx as nx
import zamalang
from mlir.dialects import builtin
from mlir.ir import Context, InsertionPoint, Location, Module

from ..operator_graph import OPGraph
from ..representation.intermediate import Input
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

        with Context() as ctx, Location.unknown():
            zamalang.register_dialects(ctx)

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

                    for node in nx.topological_sort(op_graph.graph):
                        if isinstance(node, Input):
                            continue

                        preds = [ir_to_mlir[pred] for pred in op_graph.get_ordered_preds(node)]
                        node_converter = IntermediateNodeConverter(ctx, op_graph, node, preds)
                        ir_to_mlir[node] = node_converter.convert(additional_conversion_info)

                    results = (
                        ir_to_mlir[output_node] for output_node in op_graph.get_ordered_outputs()
                    )
                    return results

        return str(module)

    @staticmethod
    @abstractmethod
    def _generate_additional_info_dict(op_graph: OPGraph) -> Dict[str, Any]:
        """Generate additional conversion info dict for the MLIR converter.

        Args:
            op_graph (OPGraph): the operation graph from which the additional info will be generated

        Returns:
            Dict[str, Any]: dict of additional conversion info
        """
