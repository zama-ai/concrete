"""
Declaration of `GraphConverter` class.
"""

# pylint: disable=no-member,no-name-in-module

from copy import deepcopy
from typing import Dict, List, Optional, cast

import concrete.lang as concretelang
import networkx as nx
import numpy as np
from mlir.dialects import builtin
from mlir.ir import Context, InsertionPoint, Location, Module

from ..dtypes import Integer, SignedInteger
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..values import ClearScalar
from .node_converter import NodeConverter
from .utils import MAXIMUM_BIT_WIDTH

# pylint: enable=no-member,no-name-in-module


class GraphConverter:
    """
    GraphConverter class, to convert computation graphs to their MLIR equivalent.
    """

    @staticmethod
    def _check_node_convertibility(graph: Graph, node: Node) -> Optional[str]:
        """
        Check node convertibility to MLIR.

        Args:
            graph (Graph):
                computation graph of the node

            node (Node):
                node to be checked

        Returns:
            Optional[str]:
                None if node is convertible to MLIR, the reason for inconvertibility otherwise
        """

        # pylint: disable=too-many-branches,too-many-return-statements,too-many-statements

        inputs = node.inputs
        output = node.output

        if node.operation == Operation.Constant:
            assert_that(len(inputs) == 0)
            if not isinstance(output.dtype, Integer):
                return "only integer constants are supported"

        elif node.operation == Operation.Input:
            assert_that(len(inputs) == 1)
            assert_that(inputs[0] == output)
            if not isinstance(output.dtype, Integer) or output.dtype.is_signed:
                return "only unsigned integer inputs are supported"

        else:
            assert_that(node.operation == Operation.Generic)

            if not isinstance(output.dtype, Integer):
                return "only integer operations are supported"

            name = node.properties["name"]

            if name == "add":
                assert_that(len(inputs) == 2)

            elif name == "concatenate":
                if not all(input.is_encrypted for input in inputs):
                    return "only all encrypted concatenate is supported"

            elif name in ["conv1d", "conv2d", "conv3d"]:
                assert_that(len(inputs) == 2 or len(inputs) == 3)
                if not (inputs[0].is_encrypted and inputs[1].is_clear):
                    return f"only {name} with encrypted input and clear weight is supported"

            elif name == "dot":
                assert_that(len(inputs) == 2)
                if inputs[0].is_encrypted and inputs[1].is_encrypted:
                    return "only dot product between encrypted and clear is supported"

            elif name == "index.static":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted indexing supported"

            elif name == "matmul":
                assert_that(len(inputs) == 2)
                if inputs[0].is_encrypted and inputs[1].is_encrypted:
                    return "only matrix multiplication between encrypted and clear is supported"

            elif name == "multiply":
                assert_that(len(inputs) == 2)
                if inputs[0].is_encrypted and inputs[1].is_encrypted:
                    return "only multiplication between encrypted and clear is supported"

            elif name == "negative":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted negation is supported"

            elif name == "ones":
                assert_that(len(inputs) == 0)

            elif name == "reshape":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted reshape is supported"

            elif name == "subtract":
                assert_that(len(inputs) == 2)
                if not (inputs[0].is_clear and inputs[1].is_encrypted):
                    return "only subtraction of encrypted from clear is supported"

            elif name == "sum":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted sum is supported"

            elif name == "transpose":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted transpose is supported"

            elif name == "zeros":
                assert_that(len(inputs) == 0)

            else:
                assert_that(node.converted_to_table_lookup)
                variable_input_indices = [
                    idx
                    for idx, pred in enumerate(graph.ordered_preds_of(node))
                    if not pred.operation == Operation.Constant
                ]
                if len(variable_input_indices) != 1:
                    return "only single input table lookups are supported"

            if len(inputs) > 0 and all(input.is_clear for input in inputs):
                return "one of the operands must be encrypted"

        return None

        # pylint: enable=too-many-branches,too-many-return-statements,too-many-statements

    @staticmethod
    def _check_graph_convertibility(graph: Graph):
        """
        Check graph convertibility to MLIR.

        Args:
            graph (Graph):
                computation graph to be checked

        Raises:
            RuntimeError:
                if `graph` is not convertible to MLIR
        """

        offending_nodes = {}

        if len(graph.output_nodes) > 1:
            offending_nodes.update(
                {
                    node: ["only a single output is supported"]
                    for node in graph.output_nodes.values()
                }
            )

        if len(offending_nodes) == 0:
            for node in graph.graph.nodes:
                if (reason := GraphConverter._check_node_convertibility(graph, node)) is not None:
                    offending_nodes[node] = [reason]

        if len(offending_nodes) != 0:
            raise RuntimeError(
                "Function you are trying to compile cannot be converted to MLIR\n\n"
                + graph.format(highlighted_nodes=offending_nodes)
            )

    @staticmethod
    def _update_bit_widths(graph: Graph):
        """
        Update bit-widths in a computation graph to be convertible to MLIR.

        Args:
            graph (Graph):
                computation graph to be updated
        """

        offending_nodes: Dict[Node, List[str]] = {}

        max_bit_width = 0
        for node in graph.graph.nodes:
            dtype = node.output.dtype
            assert_that(isinstance(dtype, Integer))

            current_node_bit_width = (
                dtype.bit_width - 1 if node.output.is_clear else dtype.bit_width
            )
            max_bit_width = max(max_bit_width, current_node_bit_width)

            if current_node_bit_width > MAXIMUM_BIT_WIDTH:
                offending_nodes[node] = [
                    f"only up to {MAXIMUM_BIT_WIDTH}-bit integers are supported"
                ]

        if len(offending_nodes) != 0:
            raise RuntimeError(
                "Function you are trying to compile cannot be converted to MLIR:\n\n"
                + graph.format(highlighted_nodes=offending_nodes)
            )

        for node in graph.graph.nodes:
            for value in node.inputs + [node.output]:
                dtype = value.dtype
                assert_that(isinstance(dtype, Integer))
                dtype.bit_width = max_bit_width + 1 if value.is_clear else max_bit_width

    @staticmethod
    def _offset_negative_lookup_table_inputs(graph: Graph):
        """
        Offset negative table lookup inputs to be convertible to MLIR.

        Args:
            graph (Graph):
                computation graph to apply offset
        """

        # ugly hack to add an offset before entering a TLU
        # if its variable input node has a signed output.
        # this makes hardcoded assumptions about the way bit widths are handled in MLIR.
        # this does not update the TLU input values to allow for proper table generation.

        nx_graph = graph.graph
        for node in list(nx_graph.nodes):
            if node.operation == Operation.Generic:
                if not node.converted_to_table_lookup:
                    continue

                variable_input_index = -1

                preds = graph.ordered_preds_of(node)
                for index, pred in enumerate(preds):
                    if pred.operation != Operation.Constant:
                        variable_input_index = index
                        break

                variable_input_node = preds[variable_input_index]

                variable_input_value = variable_input_node.output
                variable_input_dtype = variable_input_value.dtype

                assert_that(isinstance(variable_input_dtype, Integer))
                variable_input_dtype = cast(Integer, variable_input_dtype)

                if not variable_input_dtype.is_signed:
                    continue

                variable_input_bit_width = variable_input_dtype.bit_width
                offset_constant_dtype = SignedInteger(variable_input_bit_width + 1)

                offset_constant = Node.constant(abs(variable_input_dtype.min()))
                offset_constant.output.dtype = offset_constant_dtype

                add_offset = Node.generic(
                    "add",
                    [variable_input_value, ClearScalar(offset_constant_dtype)],
                    variable_input_value,
                    np.add,
                )

                nx_graph.remove_edge(variable_input_node, node)

                nx_graph.add_edge(variable_input_node, add_offset, input_idx=0)
                nx_graph.add_edge(offset_constant, add_offset, input_idx=1)

                nx_graph.add_edge(add_offset, node, input_idx=variable_input_index)

    @staticmethod
    def convert(graph: Graph, virtual: bool = False) -> str:
        """
        Convert a computation graph to its corresponding MLIR representation.

        Args:
            graph (Graph):
                computation graph to be converted

            virtual  (bool, default = False):
                whether to circuit will be virtual

        Returns:
            str:
                textual MLIR representation corresponding to `graph`
        """

        graph = deepcopy(graph)

        GraphConverter._check_graph_convertibility(graph)
        if virtual:
            return "Virtual circuits doesn't have MLIR."

        GraphConverter._update_bit_widths(graph)
        GraphConverter._offset_negative_lookup_table_inputs(graph)

        # There are no tensor +*- scalar operations in the compiler
        # But such operations are used commonly, so we need to support them
        # So, we implemented some workarounds (pull request #970)
        # Once we have native support, this workaround shall be removed (issue #837)
        # (most changes in #970 shall be reverted)

        # { node1: "%arg0", node2: "%0", node3: "%1" }
        nodes_to_mlir_names: Dict[Node, str] = {}

        # { "%arg0": "i5", "%0": "tensor<2x3x!FHE.eint<4>>" }
        mlir_names_to_mlir_types: Dict[str, str] = {}

        # { "%0": ["%c1_i5"] } == for %0 we need to convert %c1_i5 to 1d tensor
        scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]] = {}

        with Context() as ctx, Location.unknown():
            concretelang.register_dialects(ctx)

            module = Module.create()
            with InsertionPoint(module.body):
                parameters = [
                    NodeConverter.value_to_mlir_type(ctx, input_node.output)
                    for input_node in graph.ordered_inputs()
                ]

                @builtin.FuncOp.from_py_func(*parameters)
                def main(*arg):
                    ir_to_mlir = {}
                    for arg_num, node in graph.input_nodes.items():
                        ir_to_mlir[node] = arg[arg_num]

                        mlir_name = f"%arg{arg_num}"
                        nodes_to_mlir_names[node] = mlir_name
                        mlir_names_to_mlir_types[mlir_name] = str(parameters[arg_num])

                    for node in nx.topological_sort(graph.graph):
                        if node.operation == Operation.Input:
                            continue

                        preds = [ir_to_mlir[pred] for pred in graph.ordered_preds_of(node)]
                        node_converter = NodeConverter(
                            ctx,
                            graph,
                            node,
                            preds,
                            nodes_to_mlir_names,
                            mlir_names_to_mlir_types,
                            scalar_to_1d_tensor_conversion_hacks,
                        )
                        ir_to_mlir[node] = node_converter.convert()

                    results = (ir_to_mlir[output_node] for output_node in graph.ordered_outputs())
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

        return "\n".join(module_lines_after_hacks_are_applied).strip()
