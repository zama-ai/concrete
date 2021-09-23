"""File containing code to convert a DAG containing ir nodes to the compiler opset."""
# pylint: disable=no-name-in-module,no-member
from typing import Tuple, cast

import networkx as nx
import zamalang
from mlir.dialects import builtin
from mlir.ir import Context, InsertionPoint, IntegerType, Location, Module, RankedTensorType
from mlir.ir import Type as MLIRType
from mlir.ir import UnrankedTensorType
from zamalang.dialects import hlfhe

from .. import values
from ..data_types import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_scalar_unsigned_integer,
    value_is_encrypted_tensor_unsigned_integer,
)
from ..debugging.custom_assert import custom_assert
from ..operator_graph import OPGraph
from ..representation.intermediate import Input


class MLIRConverter:
    """Converter of the common IR to MLIR."""

    def __init__(self, conversion_functions: dict) -> None:
        """Instantiate a converter with a given set of converters.

        Args:
            conversion_functions (dict): mapping common IR nodes to functions that generate MLIR.
                every function should have 4 arguments:
                    - node (IntermediateNode): the node itself to be converted
                    - operands (IntermediateNode): predecessors of node ordered as operands
                    - ir_to_mlir_node (dict): mapping between IntermediateNode and their equivalent
                        MLIR values
                    - context (mlir.Context): the MLIR context being used for the conversion
        """
        self.conversion_functions = conversion_functions
        self._init_context()

    def _init_context(self):
        self.context = Context()
        zamalang.register_dialects(self.context)

    def _get_tensor_type(
        self,
        bit_width: int,
        is_encrypted: bool,
        is_signed: bool,
        shape: Tuple[int, ...],
    ) -> MLIRType:
        """Get the MLIRType for a tensor element given its properties.

        Args:
            bit_width (int): number of bits used for the scalar
            is_encrypted (bool): is the scalar encrypted or not
            is_signed (bool): is the scalar signed or not
            shape (Tuple[int, ...]): shape of the tensor

        Returns:
            MLIRType: corresponding MLIR type
        """
        element_type = self._get_scalar_integer_type(bit_width, is_encrypted, is_signed)
        if len(shape):  # randked tensor
            return RankedTensorType.get(shape, element_type)
        # unranked tensor
        return UnrankedTensorType.get(element_type)

    def _get_scalar_integer_type(
        self, bit_width: int, is_encrypted: bool, is_signed: bool
    ) -> MLIRType:
        """Get the MLIRType for a scalar element given its properties.

        Args:
            bit_width (int): number of bits used for the scalar
            is_encrypted (bool): is the scalar encrypted or not
            is_signed (bool): is the scalar signed or not

        Returns:
            MLIRType: corresponding MLIR type
        """
        if is_encrypted and not is_signed:
            return hlfhe.EncryptedIntegerType.get(self.context, bit_width)
        if is_signed and not is_encrypted:  # clear signed
            return IntegerType.get_signed(bit_width)
        # should be clear unsigned at this point
        custom_assert(not is_signed and not is_encrypted)
        # unsigned integer are considered signless in the compiler
        return IntegerType.get_signless(bit_width)

    def common_value_to_mlir_type(self, value: values.BaseValue) -> MLIRType:
        """Convert a common value to its corresponding MLIR Type.

        Args:
            value: value to convert

        Returns:
            corresponding MLIR type
        """
        if value_is_encrypted_scalar_unsigned_integer(value):
            return self._get_scalar_integer_type(
                cast(Integer, value.data_type).bit_width, True, False
            )
        if value_is_clear_scalar_integer(value):
            dtype = cast(Integer, value.data_type)
            return self._get_scalar_integer_type(
                dtype.bit_width, is_encrypted=False, is_signed=dtype.is_signed
            )
        if value_is_encrypted_tensor_unsigned_integer(value):
            dtype = cast(Integer, value.data_type)
            return self._get_tensor_type(
                dtype.bit_width,
                is_encrypted=True,
                is_signed=False,
                shape=cast(values.TensorValue, value).shape,
            )
        if value_is_clear_tensor_integer(value):
            dtype = cast(Integer, value.data_type)
            return self._get_tensor_type(
                dtype.bit_width,
                is_encrypted=False,
                is_signed=dtype.is_signed,
                shape=cast(values.TensorValue, value).shape,
            )
        raise TypeError(f"can't convert value of type {type(value)} to MLIR type")

    def convert(self, op_graph: OPGraph) -> str:
        """Convert the graph of IntermediateNode to an MLIR textual representation.

        Args:
            graph: graph of IntermediateNode to be converted

        Returns:
            textual MLIR representation
        """
        with self.context, Location.unknown():
            module = Module.create()
            # collect inputs
            with InsertionPoint(module.body):
                func_types = [
                    self.common_value_to_mlir_type(input_node.inputs[0])
                    for input_node in op_graph.get_ordered_inputs()
                ]

                @builtin.FuncOp.from_py_func(*func_types)
                def main(*arg):
                    ir_to_mlir_node = {}
                    for arg_num, node in op_graph.input_nodes.items():
                        ir_to_mlir_node[node] = arg[arg_num]
                    for node in nx.topological_sort(op_graph.graph):
                        if isinstance(node, Input):
                            continue
                        mlir_op = self.conversion_functions.get(type(node), None)
                        if mlir_op is None:  # pragma: no cover
                            raise NotImplementedError(
                                f"we don't yet support conversion to MLIR of computations using"
                                f"{type(node)}"
                            )
                        # get sorted preds: sorted by their input index
                        # replication of pred is possible (e.g lambda x: x + x)
                        idx_to_pred = {}
                        for pred in op_graph.graph.pred[node]:
                            edge_data = op_graph.graph.get_edge_data(pred, node)
                            for data in edge_data.values():
                                idx_to_pred[data["input_idx"]] = pred
                        preds = [idx_to_pred[i] for i in range(len(idx_to_pred))]
                        # convert to mlir
                        result = mlir_op(node, preds, ir_to_mlir_node, self.context)
                        ir_to_mlir_node[node] = result

                    results = (
                        ir_to_mlir_node[output_node]
                        for output_node in op_graph.get_ordered_outputs()
                    )
                    return results

        return module.__str__()


# pylint: enable=no-name-in-module,no-member
