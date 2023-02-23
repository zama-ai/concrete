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
from mlir.dialects import arith, func
from mlir.ir import (
    Attribute,
    Context,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    OpResult,
    RankedTensorType,
)

from ..dtypes import Integer, SignedInteger
from ..internal.utils import assert_that
from ..representation import Graph, Node, Operation
from ..values import ClearScalar, EncryptedScalar
from .node_converter import NodeConverter
from .utils import MAXIMUM_TLU_BIT_WIDTH

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

            elif name == "assign.static":
                if not inputs[0].is_encrypted:
                    return "only assignment to encrypted tensors are supported"

            elif name in ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"]:
                assert_that(len(inputs) == 2)
                if all(value.is_encrypted for value in node.inputs):
                    pred_nodes = graph.ordered_preds_of(node)
                    if (
                        name in ["left_shift", "right_shift"]
                        and cast(Integer, pred_nodes[1].output.dtype).bit_width > 4
                    ):
                        return "only up to 4-bit shifts are supported"

                    for pred_node in pred_nodes:
                        assert isinstance(pred_node.output.dtype, Integer)
                        if pred_node.output.dtype.is_signed:
                            return "only unsigned bitwise operations are supported"

            elif name == "broadcast_to":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted broadcasting is supported"

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

            elif name in ["equal", "greater", "greater_equal", "less", "less_equal", "not_equal"]:
                assert_that(len(inputs) == 2)

            elif name == "expand_dims":
                assert_that(len(inputs) == 1)

            elif name == "index.static":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted indexing supported"

            elif name == "matmul":
                assert_that(len(inputs) == 2)
                if inputs[0].is_encrypted and inputs[1].is_encrypted:
                    return "only matrix multiplication between encrypted and clear is supported"

            elif name == "maxpool":
                assert_that(len(inputs) == 1)
                if not inputs[0].is_encrypted:
                    return "only encrypted maxpool is supported"

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

            elif name == "squeeze":
                assert_that(len(inputs) == 1)

            elif name == "subtract":
                assert_that(len(inputs) == 2)

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
                assert_that(len(variable_input_indices) == 1)

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
                    node: ["only a single output is supported", node.location]
                    for node in graph.output_nodes.values()
                }
            )

        if len(offending_nodes) == 0:
            for node in graph.graph.nodes:
                reason = GraphConverter._check_node_convertibility(graph, node)
                if reason is not None:
                    offending_nodes[node] = [reason, node.location]

        if len(offending_nodes) != 0:
            message = (
                "Function you are trying to compile cannot be converted to MLIR\n\n"
                + graph.format(highlighted_nodes=offending_nodes)
            )
            raise RuntimeError(message)

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
        max_bit_width_node = None

        first_tlu_node = None
        first_signed_node = None

        for node in nx.lexicographical_topological_sort(graph.graph):
            dtype = node.output.dtype
            assert_that(isinstance(dtype, Integer))

            current_node_bit_width = (
                dtype.bit_width - 1 if node.output.is_clear else dtype.bit_width
            )
            if (
                all(value.is_encrypted for value in node.inputs)
                and node.operation == Operation.Generic
                and node.properties["name"]
                in [
                    "greater",
                    "greater_equal",
                    "less",
                    "less_equal",
                ]
            ):
                # implementation of these operators require at least 4 bits
                current_node_bit_width = max(current_node_bit_width, 4)

            if max_bit_width < current_node_bit_width:
                max_bit_width = current_node_bit_width
                max_bit_width_node = node

            if node.converted_to_table_lookup and first_tlu_node is None:
                first_tlu_node = node

            if dtype.is_signed and first_signed_node is None:
                first_signed_node = node

        if first_tlu_node is not None:
            if max_bit_width > MAXIMUM_TLU_BIT_WIDTH:
                assert max_bit_width_node is not None
                offending_nodes[max_bit_width_node] = [
                    (
                        {
                            Operation.Input: f"this input is {max_bit_width}-bits",
                            Operation.Constant: f"this constant is {max_bit_width}-bits",
                            Operation.Generic: f"this operation results in {max_bit_width}-bits",
                        }[max_bit_width_node.operation]
                    ),
                    max_bit_width_node.location,
                ]
                offending_nodes[first_tlu_node] = [
                    f"table lookups are only supported on circuits with "
                    f"up to {MAXIMUM_TLU_BIT_WIDTH}-bits",
                    first_tlu_node.location,
                ]

        if len(offending_nodes) != 0:
            raise RuntimeError(
                "Function you are trying to compile cannot be converted to MLIR:\n\n"
                + graph.format(highlighted_nodes=offending_nodes)
            )

        for node in nx.topological_sort(graph.graph):
            assert isinstance(node.output.dtype, Integer)
            node.properties["original_bit_width"] = node.output.dtype.bit_width

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

                offset_constant_value = abs(variable_input_dtype.min())

                offset_constant = Node.constant(offset_constant_value)
                offset_constant.output.dtype = offset_constant_dtype

                original_bit_width = Integer.that_can_represent(offset_constant_value).bit_width
                offset_constant.properties["original_bit_width"] = original_bit_width

                add_offset = Node.generic(
                    "add",
                    [variable_input_value, ClearScalar(offset_constant_dtype)],
                    variable_input_value,
                    np.add,
                )

                original_bit_width = variable_input_node.properties["original_bit_width"]
                add_offset.properties["original_bit_width"] = original_bit_width

                nx_graph.remove_edge(variable_input_node, node)

                nx_graph.add_edge(variable_input_node, add_offset, input_idx=0)
                nx_graph.add_edge(offset_constant, add_offset, input_idx=1)

                nx_graph.add_edge(add_offset, node, input_idx=variable_input_index)

    @staticmethod
    def _broadcast_assignments(graph: Graph):
        """
        Broadcast assignments.

        Args:
            graph (Graph):
                computation graph to transform
        """

        nx_graph = graph.graph
        for node in list(nx_graph.nodes):
            if node.operation == Operation.Generic and node.properties["name"] == "assign.static":
                shape = node.inputs[0].shape
                index = node.properties["kwargs"]["index"]

                assert_that(isinstance(index, tuple))
                while len(index) < len(shape):
                    index = (*index, slice(None, None, None))

                required_value_shape_list = []

                for i, indexing_element in enumerate(index):
                    if isinstance(indexing_element, slice):
                        n = len(np.zeros(shape[i])[indexing_element])
                        required_value_shape_list.append(n)
                    else:
                        required_value_shape_list.append(1)

                required_value_shape = tuple(required_value_shape_list)
                actual_value_shape = node.inputs[1].shape

                if required_value_shape != actual_value_shape:
                    preds = graph.ordered_preds_of(node)
                    pred_to_modify = preds[1]

                    modified_value = deepcopy(pred_to_modify.output)
                    modified_value.shape = required_value_shape

                    try:
                        np.broadcast_to(np.zeros(actual_value_shape), required_value_shape)
                        modified_value.is_encrypted = True
                        modified_value.dtype = node.output.dtype
                        modified_pred = Node.generic(
                            "broadcast_to",
                            [pred_to_modify.output],
                            modified_value,
                            np.broadcast_to,
                            kwargs={"shape": required_value_shape},
                        )
                    except Exception:  # pylint: disable=broad-except
                        np.reshape(np.zeros(actual_value_shape), required_value_shape)
                        modified_pred = Node.generic(
                            "reshape",
                            [pred_to_modify.output],
                            modified_value,
                            np.reshape,
                            kwargs={"newshape": required_value_shape},
                        )

                    modified_pred.properties["original_bit_width"] = pred_to_modify.properties[
                        "original_bit_width"
                    ]

                    nx_graph.add_edge(pred_to_modify, modified_pred, input_idx=0)

                    nx_graph.remove_edge(pred_to_modify, node)
                    nx_graph.add_edge(modified_pred, node, input_idx=1)

                    node.inputs[1] = modified_value

    @staticmethod
    def _encrypt_clear_assignments(graph: Graph):
        """
        Encrypt clear assignments.

        Args:
            graph (Graph):
                computation graph to transform
        """

        nx_graph = graph.graph
        for node in list(nx_graph.nodes):
            if node.operation == Operation.Generic and node.properties["name"] == "assign.static":
                assigned_value = node.inputs[1]
                if assigned_value.is_clear:
                    preds = graph.ordered_preds_of(node)
                    assigned_pred = preds[1]

                    new_assigned_pred_value = deepcopy(assigned_value)
                    new_assigned_pred_value.is_encrypted = True
                    new_assigned_pred_value.dtype = preds[0].output.dtype

                    zero = Node.generic(
                        "zeros",
                        [],
                        EncryptedScalar(new_assigned_pred_value.dtype),
                        lambda: np.zeros((), dtype=np.int64),
                    )

                    original_bit_width = 1
                    zero.properties["original_bit_width"] = original_bit_width

                    new_assigned_pred = Node.generic(
                        "add",
                        [assigned_pred.output, zero.output],
                        new_assigned_pred_value,
                        np.add,
                    )

                    original_bit_width = assigned_pred.properties["original_bit_width"]
                    new_assigned_pred.properties["original_bit_width"] = original_bit_width

                    nx_graph.remove_edge(preds[1], node)

                    nx_graph.add_edge(preds[1], new_assigned_pred, input_idx=0)
                    nx_graph.add_edge(zero, new_assigned_pred, input_idx=1)

                    nx_graph.add_edge(new_assigned_pred, node, input_idx=1)

    @staticmethod
    def _tensorize_scalars_for_fhelinalg(graph: Graph):
        """
        Tensorize scalars if they are used within fhelinalg operations.

        Args:
            graph (Graph):
                computation graph to update
        """

        # pylint: disable=invalid-name
        OPS_TO_TENSORIZE = [
            "add",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "broadcast_to",
            "dot",
            "equal",
            "greater",
            "greater_equal",
            "left_shift",
            "less",
            "less_equal",
            "multiply",
            "not_equal",
            "right_shift",
            "subtract",
        ]
        # pylint: enable=invalid-name

        tensorized_scalars: Dict[Node, Node] = {}

        nx_graph = graph.graph
        for node in list(nx_graph.nodes):
            if node.operation == Operation.Generic and node.properties["name"] in OPS_TO_TENSORIZE:
                assert len(node.inputs) in {1, 2}

                if len(node.inputs) == 2:
                    if {inp.is_scalar for inp in node.inputs} != {True, False}:
                        continue
                else:
                    if not node.inputs[0].is_scalar:
                        continue

                # for bitwise and comparison operators that can have constants
                # we don't need broadcasting here
                if node.converted_to_table_lookup:
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

                    original_bit_width = pred_to_tensorize.properties["original_bit_width"]
                    tensorized_pred.properties["original_bit_width"] = original_bit_width

                    original_shape = ()
                    tensorized_pred.properties["original_shape"] = original_shape

                    nx_graph.add_edge(pred_to_tensorize, tensorized_pred, input_idx=0)
                    tensorized_scalars[pred_to_tensorize] = tensorized_pred

                assert tensorized_pred is not None

                nx_graph.remove_edge(pred_to_tensorize, node)
                nx_graph.add_edge(tensorized_pred, node, input_idx=pred_to_tensorize_index)

                new_input_value = deepcopy(node.inputs[pred_to_tensorize_index])
                new_input_value.shape = (1,)
                node.inputs[pred_to_tensorize_index] = new_input_value

    @staticmethod
    def _sanitize_signed_inputs(graph: Graph, args: List[Any], ctx: Context) -> List[Any]:
        """
        Use subtraction to sanitize signed inputs.

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

                sanitizer_type = IntegerType.get_signless(n + 1)
                sanitizer = 2 ** (n - 1)

                if input_value.is_scalar:
                    sanitizer_attr = IntegerAttr.get(sanitizer_type, sanitizer)
                else:
                    sanitizer_type = RankedTensorType.get((1,), sanitizer_type)
                    sanitizer_attr = Attribute.parse(f"dense<[{sanitizer}]> : {sanitizer_type}")

                # pylint: disable=too-many-function-args
                sanitizer_cst = arith.ConstantOp(sanitizer_type, sanitizer_attr)
                # pylint: enable=too-many-function-args

                resulting_type = NodeConverter.value_to_mlir_type(ctx, input_value)
                if input_value.is_scalar:
                    sanitized = fhe.SubEintIntOp(resulting_type, arg, sanitizer_cst).result
                else:
                    sanitized = fhelinalg.SubEintIntOp(resulting_type, arg, sanitizer_cst).result

                sanitized_args.append(sanitized)
            else:
                sanitized_args.append(arg)

        return sanitized_args

    @staticmethod
    def convert(graph: Graph) -> str:
        """
        Convert a computation graph to its corresponding MLIR representation.

        Args:
            graph (Graph):
                computation graph to be converted

        Returns:
            str:
                textual MLIR representation corresponding to `graph`
        """

        graph = deepcopy(graph)

        GraphConverter._check_graph_convertibility(graph)
        GraphConverter._update_bit_widths(graph)
        GraphConverter._offset_negative_lookup_table_inputs(graph)
        GraphConverter._broadcast_assignments(graph)
        GraphConverter._encrypt_clear_assignments(graph)
        GraphConverter._tensorize_scalars_for_fhelinalg(graph)

        from_elements_operations: Dict[OpResult, List[OpResult]] = {}

        with Context() as ctx, Location.unknown():
            concretelang.register_dialects(ctx)

            module = Module.create()
            with InsertionPoint(module.body):
                parameters = [
                    NodeConverter.value_to_mlir_type(ctx, input_node.output)
                    for input_node in graph.ordered_inputs()
                ]

                @func.FuncOp.from_py_func(*parameters)
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
                            from_elements_operations,
                        )
                        ir_to_mlir[node] = node_converter.convert()

                    results = (ir_to_mlir[output_node] for output_node in graph.ordered_outputs())
                    return results

        direct_replacements = {}
        for placeholder, elements in from_elements_operations.items():
            element_names = [NodeConverter.mlir_name(element) for element in elements]
            actual_value = f"tensor.from_elements {', '.join(element_names)} : {placeholder.type}"
            direct_replacements[NodeConverter.mlir_name(placeholder)] = actual_value

        module_lines_after_hacks_are_applied = []
        for line in str(module).split("\n"):
            mlir_name = line.split("=")[0].strip()
            if mlir_name not in direct_replacements:
                module_lines_after_hacks_are_applied.append(line)
                continue

            new_value = direct_replacements[mlir_name]
            module_lines_after_hacks_are_applied.append(f"    {mlir_name} = {new_value}")

        return "\n".join(module_lines_after_hacks_are_applied).strip()
