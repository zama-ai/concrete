"""Converter functions from the common IR to MLIR.

Converter functions all have the same signature `converter(node, preds, ir_to_mlir_node, ctx)`
- `node`: IntermediateNode to be converted
- `preds`: List of predecessors of `node` ordered as operands
- `ir_to_mlir_node`: Dict mapping intermediate nodes to MLIR nodes or values
- `ctx`: MLIR context
"""
from typing import cast

# pylint: disable=no-name-in-module,no-member
import numpy
from mlir.dialects import arith as arith_dialect
from mlir.ir import Attribute, DenseElementsAttr, IntegerAttr, IntegerType, RankedTensorType
from zamalang.dialects import hlfhe, hlfhelinalg

from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_tensor_integer,
    value_is_encrypted_unsigned_integer,
    value_is_scalar_integer,
    value_is_tensor_integer,
)
from ..data_types.integers import Integer
from ..debugging.custom_assert import assert_true
from ..representation.intermediate import Add, Constant, Dot, GenericFunction, Mul, Sub
from ..values import TensorValue


def _convert_scalar_constant_op_to_single_element_tensor_constant_op(operation):
    """Convert a scalar constant operation result to a dense tensor constant operation result.

    see https://github.com/zama-ai/concretefhe-internal/issues/837.

    This is a temporary workaround before the compiler natively supports
    `tensor + scalar`, `tensor - scalar`, `tensor * scalar` operations.

    Example input  = `%c3_i4 = arith.constant 3 : i4`
    Example output = `%cst = arith.constant dense<3> : tensor<1xi4>`

    Args:
        operation: operation to convert

    Returns:
        the converted operation
    """

    operation_str = str(operation)

    constant_start_location = operation_str.find("arith.constant") + len("arith.constant") + 1
    constant_end_location = operation_str.find(f": {str(operation.type)}") - 1
    constant_value = operation_str[constant_start_location:constant_end_location]

    resulting_type = RankedTensorType.get((1,), operation.type)
    value_attr = Attribute.parse(f"dense<{constant_value}> : tensor<1x{str(operation.type)}>")

    return arith_dialect.ConstantOp(resulting_type, value_attr).result


def add(node, preds, ir_to_mlir_node, ctx, _additional_conversion_info=None):
    """Convert an addition intermediate node."""
    assert_true(len(node.inputs) == 2, "addition should have two inputs")
    assert_true(len(node.outputs) == 1, "addition should have a single output")

    is_convertible = True
    one_of_the_inputs_is_a_tensor = False
    both_of_the_inputs_are_encrypted = True
    ordered_preds = preds

    for input_ in node.inputs:
        if value_is_tensor_integer(input_):
            one_of_the_inputs_is_a_tensor = True
        elif not value_is_scalar_integer(input_):
            is_convertible = False

    if not is_convertible:
        raise TypeError(
            f"Don't support addition between {str(node.inputs[0])} and {str(node.inputs[1])}"
        )

    if node.inputs[1].is_clear:
        both_of_the_inputs_are_encrypted = False
    if node.inputs[0].is_clear:
        both_of_the_inputs_are_encrypted = False
        ordered_preds = preds[::-1]

    if one_of_the_inputs_is_a_tensor:
        if both_of_the_inputs_are_encrypted:
            return _linalg_add_eint_eint(node, ordered_preds, ir_to_mlir_node, ctx)
        return _linalg_add_eint_int(node, ordered_preds, ir_to_mlir_node, ctx)

    if both_of_the_inputs_are_encrypted:
        return _add_eint_eint(node, ordered_preds, ir_to_mlir_node, ctx)
    return _add_eint_int(node, ordered_preds, ir_to_mlir_node, ctx)


def _add_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width),
        lhs,
        rhs,
    ).result


def _add_eint_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate node with (eint, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width),
        lhs,
        rhs,
    ).result


def _linalg_add_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate tensor node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]

    if not str(rhs.type).startswith("tensor"):
        rhs = _convert_scalar_constant_op_to_single_element_tensor_constant_op(rhs)

    int_type = hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width)
    vec_type = RankedTensorType.get(node.outputs[0].shape, int_type)

    return hlfhelinalg.AddEintIntOp(vec_type, lhs, rhs).result


def _linalg_add_eint_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate tensor node with (eint, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]

    int_type = hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width)
    vec_type = RankedTensorType.get(node.outputs[0].shape, int_type)

    return hlfhelinalg.AddEintOp(vec_type, lhs, rhs).result


def sub(node, preds, ir_to_mlir_node, ctx, _additional_conversion_info=None):
    """Convert a subtraction intermediate node."""
    assert_true(len(node.inputs) == 2, "subtraction should have two inputs")
    assert_true(len(node.outputs) == 1, "subtraction should have a single output")

    is_convertible = True
    one_of_the_inputs_is_a_tensor = False

    if value_is_clear_tensor_integer(node.inputs[0]):
        one_of_the_inputs_is_a_tensor = True
    elif not value_is_clear_scalar_integer(node.inputs[0]):
        is_convertible = False

    if value_is_tensor_integer(node.inputs[1]):
        one_of_the_inputs_is_a_tensor = True
    elif not value_is_scalar_integer(node.inputs[1]):
        is_convertible = False

    if not is_convertible:
        raise TypeError(
            f"Don't support subtraction between {str(node.inputs[0])} and {str(node.inputs[1])}"
        )

    if one_of_the_inputs_is_a_tensor:
        return _linalg_sub_int_eint(node, preds, ir_to_mlir_node, ctx)
    return _sub_int_eint(node, preds, ir_to_mlir_node, ctx)


def _sub_int_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert a subtraction intermediate node with (int, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.SubIntEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width),
        lhs,
        rhs,
    ).result


def _linalg_sub_int_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert a subtraction intermediate node with (int, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]

    if not str(lhs.type).startswith("tensor"):
        lhs = _convert_scalar_constant_op_to_single_element_tensor_constant_op(lhs)

    int_type = hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width)
    vec_type = RankedTensorType.get(node.outputs[0].shape, int_type)

    return hlfhelinalg.SubIntEintOp(vec_type, lhs, rhs).result


def mul(node, preds, ir_to_mlir_node, ctx, _additional_conversion_info=None):
    """Convert a multiplication intermediate node."""
    assert_true(len(node.inputs) == 2, "multiplication should have two inputs")
    assert_true(len(node.outputs) == 1, "multiplication should have a single output")

    is_convertible = True
    one_of_the_inputs_is_a_tensor = False
    ordered_preds = preds

    for input_ in node.inputs:
        if value_is_tensor_integer(input_):
            one_of_the_inputs_is_a_tensor = True
        elif not value_is_scalar_integer(input_):
            is_convertible = False

    if not is_convertible:
        raise TypeError(
            f"Don't support multiplication between {str(node.inputs[0])} and {str(node.inputs[1])}"
        )

    if node.inputs[0].is_clear:
        ordered_preds = preds[::-1]

    if one_of_the_inputs_is_a_tensor:
        return _linalg_mul_eint_int(node, ordered_preds, ir_to_mlir_node, ctx)
    return _mul_eint_int(node, ordered_preds, ir_to_mlir_node, ctx)


def _mul_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert a multiplication intermediate node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.MulEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width),
        lhs,
        rhs,
    ).result


def _linalg_mul_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert a subtraction intermediate node with (int, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]

    if not str(rhs.type).startswith("tensor"):
        rhs = _convert_scalar_constant_op_to_single_element_tensor_constant_op(rhs)

    int_type = hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width)
    vec_type = RankedTensorType.get(node.outputs[0].shape, int_type)

    return hlfhelinalg.MulEintIntOp(vec_type, lhs, rhs).result


def constant(node, _preds, _ir_to_mlir_node, ctx, _additional_conversion_info=None):
    """Convert a constant input."""
    value = node.outputs[0]

    if value_is_clear_scalar_integer(value):
        value = cast(TensorValue, value)

        dtype = cast(Integer, value.dtype)
        data = node.constant_data

        int_type = IntegerType.get_signless(dtype.bit_width, context=ctx)
        return arith_dialect.ConstantOp(int_type, IntegerAttr.get(int_type, data)).result

    if value_is_clear_tensor_integer(value):
        value = cast(TensorValue, value)

        dtype = cast(Integer, value.dtype)
        data = node.constant_data

        int_type = IntegerType.get_signless(dtype.bit_width, context=ctx)
        vec_type = RankedTensorType.get(value.shape, int_type)

        # usage of `Attribute.parse` is the result of some limitations in the MLIR module
        # provided by LLVM

        # `DenseElementsAttr` should have been used instead but it's impossible to assign
        # custom bit-widths using it (e.g., uint5)

        # since we coudn't create a `DenseElementsAttr` with a custom bit width using python api
        # we use `Attribute.parse` to let the underlying library do it by itself

        value_attr = Attribute.parse(f"dense<{str(data.tolist())}> : {vec_type}")
        return arith_dialect.ConstantOp(vec_type, value_attr).result

    raise TypeError(f"Don't support {value} constants")


def apply_lut(node, preds, ir_to_mlir_node, ctx, additional_conversion_info):
    """Convert a GenericFunction intermediate node."""

    variable_input_indices = [
        idx for idx, pred in enumerate(preds) if not isinstance(pred, Constant)
    ]

    assert_true(
        (non_constant_pred_count := len(variable_input_indices)) == 1,
        f"LUT should have a single variable input (got {non_constant_pred_count})",
    )

    variable_input_idx = variable_input_indices[0]
    variable_input_value = node.inputs[variable_input_idx]

    assert_true(len(node.outputs) == 1, "LUT should have a single output")
    if not value_is_encrypted_unsigned_integer(variable_input_value):
        raise TypeError(
            f"Only support LUT with encrypted unsigned integers inputs "
            f"(but {variable_input_value} is provided)"
        )
    if not value_is_encrypted_unsigned_integer(node.outputs[0]):
        raise TypeError(
            f"Only support LUT with encrypted unsigned integers outputs "
            f"(but {node.outputs[0]} is provided)"
        )

    x_node = preds[variable_input_idx]
    x = ir_to_mlir_node[x_node]
    tables = additional_conversion_info["tables"][node]

    # TODO: #559 adapt the code to support multi TLUs
    # This cannot be reached today as compilation fails if the intermediate values are not all
    # scalars
    if len(tables) > 1:  # pragma: no cover
        raise RuntimeError(
            "MLIR conversion currently does not support multiple test vectors for LUT"
        )

    table = tables[0][0]

    out_dtype = cast(Integer, node.outputs[0].dtype)
    # Create table
    dense_elem = DenseElementsAttr.get(numpy.array(table, dtype=numpy.uint64), context=ctx)
    tensor_lut = arith_dialect.ConstantOp(
        RankedTensorType.get([len(table)], IntegerType.get_signless(64, context=ctx)),
        dense_elem,
    ).result

    int_type = hlfhe.EncryptedIntegerType.get(ctx, out_dtype.bit_width)

    if value_is_encrypted_tensor_integer(node.inputs[0]):
        vec_type = RankedTensorType.get(node.outputs[0].shape, int_type)
        return hlfhelinalg.ApplyLookupTableEintOp(vec_type, x, tensor_lut).result
    return hlfhe.ApplyLookupTableEintOp(int_type, x, tensor_lut).result


def dot(node, preds, ir_to_mlir_node, ctx, _additional_conversion_info=None):
    """Convert a dot intermediate node."""
    assert_true(len(node.inputs) == 2, "Dot should have two inputs")
    assert_true(len(node.outputs) == 1, "Dot should have a single output")
    if not (
        (
            value_is_encrypted_tensor_integer(node.inputs[0])
            and value_is_clear_tensor_integer(node.inputs[1])
        )
        or (
            value_is_encrypted_tensor_integer(node.inputs[1])
            and value_is_clear_tensor_integer(node.inputs[0])
        )
    ):
        raise TypeError(
            f"Don't support dot between {str(node.inputs[0])} and {str(node.inputs[1])}"
        )
    lhs_node, rhs_node = preds
    # need to flip as underlying operation need encrypted first
    if value_is_clear_tensor_integer(node.inputs[0]):
        lhs_node, rhs_node = rhs_node, lhs_node
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhelinalg.Dot(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].dtype.bit_width),
        lhs,
        rhs,
    ).result


V0_OPSET_CONVERSION_FUNCTIONS = {
    Add: add,
    Sub: sub,
    Mul: mul,
    Constant: constant,
    GenericFunction: apply_lut,
    Dot: dot,
}

# pylint: enable=no-name-in-module,no-member
