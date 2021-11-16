"""Module that provides IntermediateNode conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from typing import Any, Dict, List, cast

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

    def __init__(
        self, ctx: Context, op_graph: OPGraph, node: IntermediateNode, preds: List[OpResult]
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

    def convert(self, additional_conversion_info: Dict[str, Any]) -> OpResult:
        """Convert an intermediate node to its corresponding MLIR representation.

        Args:
            additional_conversion_info (Dict[str, Any]):
                external info that the converted node might need

        Returns:
            str: textual MLIR representation corresponding to self.node
        """

        if isinstance(self.node, Add):
            return self.convert_add()

        if isinstance(self.node, Constant):
            return self.convert_constant()

        if isinstance(self.node, Dot):
            return self.convert_dot()

        if isinstance(self.node, GenericFunction):
            return self.convert_generic_function(additional_conversion_info)

        if isinstance(self.node, Mul):
            return self.convert_mul()

        if isinstance(self.node, Sub):
            return self.convert_sub()

        # this statement is not covered as unsupported opeations fail on check mlir compatibility
        raise NotImplementedError(
            f"{type(self.node)} nodes cannot be converted to MLIR yet"
        )  # pragma: no cover

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

        value = self.node.inputs[variable_input_index]
        assert_true(value.is_encrypted)

        if not isinstance(value.dtype, Integer) or value.dtype.is_signed:  # pragma: no cover
            # this branch is not covered as it's impossible to get into due to how compilation works
            # however, it doesn't hurt to keep it as an extra measure
            raise NotImplementedError(f"Table lookup on {value} cannot be converted to MLIR yet")

        tables = additional_conversion_info["tables"][self.node]

        # TODO: #559 adapt the code to support multi TLUs
        # This cannot be reached today as compilation fails
        # if the intermediate values are not all scalars
        if len(tables) > 1:  # pragma: no cover
            raise NotImplementedError("Multi table lookups cannot be converted to MLIR yet")

        table = tables[0][0]

        lut_size = len(table)
        lut_type = RankedTensorType.get([lut_size], IntegerType.get_signless(64, context=self.ctx))
        lut_attr = DenseElementsAttr.get(numpy.array(table, dtype=numpy.uint64), context=self.ctx)
        lut = arith.ConstantOp(lut_type, lut_attr).result

        resulting_type = value_to_mlir_type(self.ctx, self.node.outputs[0])
        pred = self.preds[variable_input_index]

        if self.one_of_the_inputs_is_a_tensor:
            result = hlfhelinalg.ApplyLookupTableEintOp(resulting_type, pred, lut).result
        else:
            result = hlfhe.ApplyLookupTableEintOp(resulting_type, pred, lut).result

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
