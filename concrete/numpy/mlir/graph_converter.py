"""
Declaration of `GraphConverter` class.
"""

# pylint: disable=no-member,no-name-in-module

from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

import concrete.lang as concretelang
import networkx as nx
import numpy as np
from concrete.lang.dialects import fhe, fhelinalg
from mlir.dialects import arith, builtin
from mlir.ir import (
    Context,
    DenseElementsAttr,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    RankedTensorType,
)

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
            if not isinstance(output.dtype, Integer):
                return "only integer inputs are supported"
            if output.dtype.is_signed and output.is_clear:
                return "only encrypted signed integer inputs are supported"

        else:
            assert_that(node.operation == Operation.Generic)

            if not isinstance(output.dtype, Integer):
                return "only integer operations are supported"

            name = node.properties["name"]

            if name == "add":
                assert_that(len(inputs) == 2)

            elif name == "array":
                assert_that(len(inputs) > 0)
                assert_that(all(input.is_scalar for input in inputs))

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
    def _tensorize_scalars_for_fhelinalg(graph: Graph):
        """
        Tensorize scalars if they are used within fhelinalg operations.

        Args:
            graph (Graph):
                computation graph to update
        """

        # pylint: disable=invalid-name
        OPS_TO_TENSORIZE = ["add", "dot", "multiply", "subtract"]
        # pylint: enable=invalid-name

        tensorized_scalars: Dict[Node, Node] = {}

        nx_graph = graph.graph
        for node in list(nx_graph.nodes):
            if node.operation == Operation.Generic and node.properties["name"] in OPS_TO_TENSORIZE:
                assert_that(len(node.inputs) == 2)

                if set(inp.is_scalar for inp in node.inputs) != {True, False}:
                    continue

                pred_to_tensorize: Optional[Node] = None
                pred_to_tensorize_index = 0

                preds = graph.ordered_preds_of(node)
                for index, pred in enumerate(preds):
                    if pred.output.is_scalar:
                        pred_to_tensorize = pred
                        pred_to_tensorize_index = index
                        break

                assert pred_to_tensorize is not None

                tensorized_pred = tensorized_scalars.get(pred_to_tensorize)
                if tensorized_pred is None:
                    tensorized_value = deepcopy(pred_to_tensorize.output)
                    tensorized_value.shape = (1,)

                    tensorized_pred = Node.generic(
                        "array",
                        [pred_to_tensorize.output],
                        tensorized_value,
                        lambda *args: np.array(args),
                    )
                    nx_graph.add_edge(pred_to_tensorize, tensorized_pred, input_idx=0)

                    tensorized_scalars[pred_to_tensorize] = tensorized_pred

                assert tensorized_pred is not None

                nx_graph.remove_edge(pred_to_tensorize, node)
                nx_graph.add_edge(tensorized_pred, node, input_idx=pred_to_tensorize_index)

    @staticmethod
    def _sanitize_signed_inputs(graph: Graph, args: List[Any], ctx: Context) -> List[Any]:
        """
        Apply table lookup to signed inputs in the beginning of evaluation to sanitize them.

        Sanitization in this context means to apply a table lookup to obtain proper input values.

        "encrypt" method of "Client" class will convert negative inputs to their corresponding
        unsigned value in 2s complement representation.

        Here is an example for 3 bits:
        000 = 0 represents 0
        001 = 1 represents 1
        010 = 2 represents 2
        011 = 3 represents 3
        100 = 4 represents -4
        101 = 5 represents -3
        110 = 6 represents -2
        111 = 7 represents -1

        And, the following table lookup is applied before anything else to sanitize the inputs:
        [0, 1, 2, 3, -4, -3, -2, -1]

        Args:
            graph (Graph):
                computation graph being converted

            args (List[Any]):
                list of arguments from mlir main

            ctx (Context):
                mlir context where the conversion is being performed

        Returns:
            Tuple[List[str], List[Any]]:
                sanitized args and name of the sanitized variables in MLIR
        """

        sanitized_args = []
        for i, arg in enumerate(args):
            input_node = graph.input_nodes[i]
            input_value = input_node.output

            assert_that(isinstance(input_value.dtype, Integer))
            input_dtype = cast(Integer, input_value.dtype)

            if input_dtype.is_signed:
                assert_that(input_value.is_encrypted)

                n = input_dtype.bit_width
                lut_range = np.arange(2**n)

                lut_values = np.where(lut_range < (2 ** (n - 1)), lut_range, lut_range - (2**n))
                lut_type = RankedTensorType.get(
                    (2**n,), IntegerType.get_signless(64, context=ctx)
                )
                lut_attr = DenseElementsAttr.get(lut_values, context=ctx)
                lut = arith.ConstantOp(lut_type, lut_attr).result

                resulting_type = NodeConverter.value_to_mlir_type(ctx, input_value)
                if input_value.is_scalar:
                    sanitized = fhe.ApplyLookupTableEintOp(resulting_type, arg, lut).result
                else:
                    sanitized = fhelinalg.ApplyLookupTableEintOp(resulting_type, arg, lut).result

                sanitized_args.append(sanitized)
            else:
                sanitized_args.append(arg)

        return sanitized_args

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
        GraphConverter._tensorize_scalars_for_fhelinalg(graph)

        # { "%0": "tensor.from_elements ..." } == we need to convert the part after "=" for %0
        direct_replacements: Dict[str, str] = {}

        with Context() as ctx, Location.unknown():
            concretelang.register_dialects(ctx)

            module = Module.create()
            with InsertionPoint(module.body):
                parameters = [
                    NodeConverter.value_to_mlir_type(ctx, input_node.output)
                    for input_node in graph.ordered_inputs()
                ]

                @builtin.FuncOp.from_py_func(*parameters)
                def main(*args):
                    sanitized_args = GraphConverter._sanitize_signed_inputs(graph, args, ctx)

                    ir_to_mlir = {}
                    for arg_num, node in graph.input_nodes.items():
                        ir_to_mlir[node] = sanitized_args[arg_num]

                    constant_cache = {}
                    for node in nx.topological_sort(graph.graph):
                        if node.operation == Operation.Input:
                            continue

                        preds = [ir_to_mlir[pred] for pred in graph.ordered_preds_of(node)]
                        node_converter = NodeConverter(
                            ctx,
                            graph,
                            node,
                            preds,
                            constant_cache,
                            direct_replacements,
                        )
                        ir_to_mlir[node] = node_converter.convert()

                    results = (ir_to_mlir[output_node] for output_node in graph.ordered_outputs())
                    return results

        module_lines_after_hacks_are_applied = []
        for line in str(module).split("\n"):
            mlir_name = line.split("=")[0].strip()
            if mlir_name not in direct_replacements:
                module_lines_after_hacks_are_applied.append(line)
                continue

            new_value = direct_replacements[mlir_name]
            module_lines_after_hacks_are_applied.append(f"    {mlir_name} = {new_value}")

        return "\n".join(module_lines_after_hacks_are_applied).strip()
