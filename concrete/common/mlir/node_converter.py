"""Module that provides IntermediateNode conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from typing import Any, Dict, List, Tuple, cast

import numpy
from mlir.dialects import arith
from mlir.ir import (
    Attribute,
    Context,
    DenseElementsAttr,
    IntegerAttr,
    IntegerType,
    OpResult,
    RankedTensorType,
)
from zamalang.dialects import hlfhe, hlfhelinalg

from ..data_types import Integer
from ..debugging import assert_true
from ..operator_graph import OPGraph
from ..representation.intermediate import (
    Add,
    Constant,
    Dot,
    GenericFunction,
    IntermediateNode,
    MatMul,
    Mul,
    Sub,
)
from ..values import TensorValue
from .conversion_helpers import value_to_mlir_type

# pylint: enable=no-name-in-module


class IntermediateNodeConverter:
    """Converter of IntermediateNode to MLIR."""

    ctx: Context
    op_graph: OPGraph
    node: IntermediateNode
    preds: List[OpResult]

    all_of_the_inputs_are_encrypted: bool
    all_of_the_inputs_are_tensors: bool
    one_of_the_inputs_is_a_tensor: bool

    nodes_to_mlir_names: Dict[IntermediateNode, str]
    mlir_names_to_mlir_types: Dict[str, str]
    scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]]

    def __init__(
        self,
        ctx: Context,
        op_graph: OPGraph,
        node: IntermediateNode,
        preds: List[OpResult],
        nodes_to_mlir_names: Dict[OpResult, str],
        mlir_names_to_mlir_types: Dict[str, str],
        scalar_to_1d_tensor_conversion_hacks: Dict[str, List[str]],
    ):
        self.ctx = ctx
        self.op_graph = op_graph
        self.node = node
        self.preds = preds

        self.all_of_the_inputs_are_encrypted = True
        self.all_of_the_inputs_are_tensors = True
        self.one_of_the_inputs_is_a_tensor = False

        for inp in node.inputs:
            if inp.is_clear:
                self.all_of_the_inputs_are_encrypted = False

            if isinstance(inp, TensorValue):
                if inp.is_scalar:
                    self.all_of_the_inputs_are_tensors = False
                else:
                    self.one_of_the_inputs_is_a_tensor = True
            else:  # pragma: no cover
                # this branch is not covered as there are only TensorValues for now
                self.all_of_the_inputs_are_tensors = False

        self.nodes_to_mlir_names = nodes_to_mlir_names
        self.mlir_names_to_mlir_types = mlir_names_to_mlir_types
        self.scalar_to_1d_tensor_conversion_hacks = scalar_to_1d_tensor_conversion_hacks

    def convert(self, additional_conversion_info: Dict[str, Any]) -> OpResult:
        """Convert an intermediate node to its corresponding MLIR representation.

        Args:
            additional_conversion_info (Dict[str, Any]):
                external info that the converted node might need

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        if isinstance(self.node, Add):
            result = self.convert_add()

        elif isinstance(self.node, Constant):
            result = self.convert_constant()

        elif isinstance(self.node, Dot):
            result = self.convert_dot()

        elif isinstance(self.node, GenericFunction):
            result = self.convert_generic_function(additional_conversion_info)

        elif isinstance(self.node, MatMul):
            result = self.convert_matmul()

        elif isinstance(self.node, Mul):
            result = self.convert_mul()

        elif isinstance(self.node, Sub):
            result = self.convert_sub()

        else:  # pragma: no cover
            # this branch is not covered as unsupported opeations fail on check mlir compatibility
            raise NotImplementedError(f"{type(self.node)} nodes cannot be converted to MLIR yet")

        mlir_name = str(result).replace("Value(", "").split("=", maxsplit=1)[0].strip()

        self.nodes_to_mlir_names[self.node] = mlir_name
        self.mlir_names_to_mlir_types[mlir_name] = str(result.type)

        if isinstance(self.node, (Add, Mul, Sub, Dot)):
            if self.one_of_the_inputs_is_a_tensor and not self.all_of_the_inputs_are_tensors:
                to_be_converted = []
                for (pred, output) in self.op_graph.get_ordered_preds_and_inputs_of(self.node):
                    inp = pred.outputs[output]
                    if isinstance(inp, TensorValue) and inp.is_scalar:
                        to_be_converted.append(self.nodes_to_mlir_names[pred])
                self.scalar_to_1d_tensor_conversion_hacks[mlir_name] = to_be_converted

        return result

    def convert_add(self) -> OpResult:
        """Convert an Add node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.all_of_the_inputs_are_encrypted:
            if self.one_of_the_inputs_is_a_tensor:
                result = hlfhelinalg.AddEintOp(resulting_type, *preds).result
            else:
                result = hlfhe.AddEintOp(resulting_type, *preds).result
        else:
            if self.node.inputs[0].is_clear:  # pragma: no cover
                # this branch is not covered as it's impossible to get into due to how tracing works
                # however, it doesn't hurt to keep it as an extra measure
                preds = preds[::-1]

            if self.one_of_the_inputs_is_a_tensor:
                result = hlfhelinalg.AddEintIntOp(resulting_type, *preds).result
            else:
                result = hlfhe.AddEintIntOp(resulting_type, *preds).result

        return result

    def convert_constant(self) -> OpResult:
        """Convert a Constant node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 0)
        assert_true(len(self.node.outputs) == 1)

        value = self.node.outputs[0]
        if not isinstance(value, TensorValue):  # pragma: no cover
            # this branch is not covered as there are only TensorValues for now
            raise NotImplementedError(f"{value} constants cannot be converted to MLIR yet")

        resulting_type = value_to_mlir_type(self.ctx, value)
        data = cast(Constant, self.node).constant_data

        if value.is_scalar:
            attr = IntegerAttr.get(resulting_type, data)
        else:
            # usage of `Attribute.parse` is the result of some limitations in the MLIR module
            # provided by LLVM

            # what should have been used is `DenseElementsAttr` but it's impossible to assign
            # custom bit-widths using it (e.g., uint5)

            # since we coudn't create a `DenseElementsAttr` with a custom bit width using python api
            # we use `Attribute.parse` to let the underlying library do it by itself

            attr = Attribute.parse(f"dense<{str(data.tolist())}> : {resulting_type}")

        return arith.ConstantOp(resulting_type, attr).result

    def convert_dot(self) -> OpResult:
        """Convert a Dot node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]
            raise NotImplementedError(
                f"Dot product between {lhs} and {rhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.node.inputs[0].is_clear:
            preds = preds[::-1]

        if self.all_of_the_inputs_are_tensors:
            # numpy.dot(x, y) where x and y are both vectors = regular dot product
            result = hlfhelinalg.Dot(resulting_type, *preds).result

        elif not self.one_of_the_inputs_is_a_tensor:
            # numpy.dot(x, y) where x and y are both scalars = x * y
            result = hlfhe.MulEintIntOp(resulting_type, *preds).result

        else:
            # numpy.dot(x, y) where one of x or y is a scalar and the other one is a vector = x * y
            result = hlfhelinalg.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_generic_function(self, additional_conversion_info: Dict[str, Any]) -> OpResult:
        """Convert a GenericFunction node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        variable_input_indices = [
            idx
            for idx, inp in enumerate(self.op_graph.get_ordered_preds(self.node))
            if not isinstance(inp, Constant)
        ]
        if len(variable_input_indices) != 1:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how tracing works
            # however, it doesn't hurt to keep it as an extra measure
            raise NotImplementedError(
                "Table lookups with more than one variable input cannot be converted to MLIR yet"
            )
        variable_input_index = variable_input_indices[0]

        assert_true(len(self.node.outputs) == 1)
        output = self.node.outputs[0]

        value = self.node.inputs[variable_input_index]
        assert_true(value.is_encrypted)

        if not isinstance(value.dtype, Integer) or value.dtype.is_signed:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how compilation works
            # however, it doesn't hurt to keep it as an extra measure
            raise NotImplementedError(f"Table lookup on {value} cannot be converted to MLIR yet")

        tables = additional_conversion_info["tables"][self.node]
        assert_true(len(tables) > 0)

        if len(tables) == 1:
            table = tables[0][0]

            lut_shape: Tuple[int, ...] = (len(table),)

            # The reduction on 63b is to avoid problems like doing a TLU of
            # the form T[j] = 2<<j, for j which is supposed to be 7b as per
            # constraint of the compiler, while in practice, it is a small
            # value. Reducing on 64b was not ok for some reason
            lut_values = numpy.array(table % (2 << 63), dtype=numpy.uint64)
        else:
            assert_true(isinstance(output, TensorValue))
            assert isinstance(output, TensorValue)

            individual_table_size = len(tables[0][0])
            lut_shape = (*output.shape, individual_table_size)

            lut_values = numpy.zeros(lut_shape, dtype=numpy.uint64)
            for table, indices in tables:
                assert_true(len(table) == individual_table_size)
                for index in indices:
                    index = (*index, slice(None, None, 1))
                    lut_values[index] = table

        lut_type = RankedTensorType.get(lut_shape, IntegerType.get_signless(64, context=self.ctx))
        lut_attr = DenseElementsAttr.get(lut_values, context=self.ctx)
        lut = arith.ConstantOp(lut_type, lut_attr).result

        resulting_type = value_to_mlir_type(self.ctx, output)
        pred = self.preds[variable_input_index]

        if self.one_of_the_inputs_is_a_tensor:
            if len(tables) == 1:
                result = hlfhelinalg.ApplyLookupTableEintOp(resulting_type, pred, lut).result
            else:
                result = hlfhelinalg.ApplyMultiLookupTableEintOp(resulting_type, pred, lut).result
        else:
            result = hlfhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result

        return result

    def convert_matmul(self) -> OpResult:
        """Convert a MatMul node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted or self.node.inputs[0].is_clear:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]

            additional_error_info = (
                " (notice the encrypted value is in the right hand side which is not supported)"
                if self.node.inputs[0].is_clear
                else ""
            )
            raise NotImplementedError(
                f"Matrix multiplication between {lhs} and {rhs} cannot be converted to MLIR yet"
                f"{additional_error_info}",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        result = hlfhelinalg.MatMulEintIntOp(resulting_type, *preds).result

        return result

    def convert_mul(self) -> OpResult:
        """Convert a Mul node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        if self.all_of_the_inputs_are_encrypted:
            lhs = self.node.inputs[0]
            rhs = self.node.inputs[1]
            raise NotImplementedError(
                f"Multiplication between {lhs} and {rhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.node.inputs[0].is_clear:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how tracing works
            # however, it doesn't hurt to keep it as an extra measure
            preds = preds[::-1]

        if self.one_of_the_inputs_is_a_tensor:
            result = hlfhelinalg.MulEintIntOp(resulting_type, *preds).result
        else:
            result = hlfhe.MulEintIntOp(resulting_type, *preds).result

        return result

    def convert_sub(self) -> OpResult:
        """Convert a Sub node to its corresponding MLIR representation.

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        assert_true(len(self.node.inputs) == 2)
        assert_true(len(self.node.outputs) == 1)

        lhs = self.node.inputs[0]
        rhs = self.node.inputs[1]
        if not (lhs.is_clear and rhs.is_encrypted):
            raise NotImplementedError(
                f"Subtraction of {rhs} from {lhs} cannot be converted to MLIR yet",
            )

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        preds = self.preds

        if self.one_of_the_inputs_is_a_tensor:
            result = hlfhelinalg.SubIntEintOp(resulting_type, *preds).result
        else:
            result = hlfhe.SubIntEintOp(resulting_type, *preds).result

        return result
