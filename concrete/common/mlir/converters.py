"""Converter functions from the common IR to MLIR.

Converter functions all have the same signature `converter(node, preds, ir_to_mlir_node, ctx)`
- `node`: IntermediateNode to be converted
- `preds`: List of predecessors of `node` ordered as operands
- `ir_to_mlir_node`: Dict mapping intermediate nodes to MLIR nodes or values
- `ctx`: MLIR context
"""
from typing import cast

# pylint: disable=no-name-in-module,no-member
import numpy as np
from mlir.dialects import std as std_dialect
from mlir.ir import DenseElementsAttr, IntegerAttr, IntegerType, RankedTensorType
from zamalang.dialects import hlfhe

from ...common.data_types.integers import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_scalar_integer,
    value_is_clear_tensor_integer,
    value_is_encrypted_scalar_unsigned_integer,
    value_is_encrypted_tensor_integer,
)
from ..debugging.custom_assert import custom_assert
from ..representation import intermediate as ir


def add(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate node."""
    custom_assert(len(node.inputs) == 2, "addition should have two inputs")
    custom_assert(len(node.outputs) == 1, "addition should have a single output")
    if value_is_encrypted_scalar_unsigned_integer(node.inputs[0]) and value_is_clear_scalar_integer(
        node.inputs[1]
    ):
        return _add_eint_int(node, preds, ir_to_mlir_node, ctx)
    if value_is_encrypted_scalar_unsigned_integer(node.inputs[1]) and value_is_clear_scalar_integer(
        node.inputs[0]
    ):
        # flip lhs and rhs
        return _add_eint_int(node, preds[::-1], ir_to_mlir_node, ctx)
    if value_is_encrypted_scalar_unsigned_integer(
        node.inputs[0]
    ) and value_is_encrypted_scalar_unsigned_integer(node.inputs[1]):
        return _add_eint_eint(node, preds, ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support addition between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _add_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def _add_eint_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert an addition intermediate node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def sub(node, preds, ir_to_mlir_node, ctx):
    """Convert a subtraction intermediate node."""
    custom_assert(len(node.inputs) == 2, "subtraction should have two inputs")
    custom_assert(len(node.outputs) == 1, "subtraction should have a single output")
    if value_is_clear_scalar_integer(node.inputs[0]) and value_is_encrypted_scalar_unsigned_integer(
        node.inputs[1]
    ):
        return _sub_int_eint(node, preds, ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support subtraction between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _sub_int_eint(node, preds, ir_to_mlir_node, ctx):
    """Convert a subtraction intermediate node with (int, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.SubIntEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def mul(node, preds, ir_to_mlir_node, ctx):
    """Convert a multiplication intermediate node."""
    custom_assert(len(node.inputs) == 2, "multiplication should have two inputs")
    custom_assert(len(node.outputs) == 1, "multiplication should have a single output")
    if value_is_encrypted_scalar_unsigned_integer(node.inputs[0]) and value_is_clear_scalar_integer(
        node.inputs[1]
    ):
        return _mul_eint_int(node, preds, ir_to_mlir_node, ctx)
    if value_is_encrypted_scalar_unsigned_integer(node.inputs[1]) and value_is_clear_scalar_integer(
        node.inputs[0]
    ):
        # flip lhs and rhs
        return _mul_eint_int(node, preds[::-1], ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support multiplication between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _mul_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Convert a multiplication intermediate node with (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.MulEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def constant(node, _, __, ctx):
    """Convert a constant inputs."""
    if not value_is_clear_scalar_integer(node.outputs[0]):
        raise TypeError("Don't support non-integer constants")
    dtype = cast(Integer, node.outputs[0].data_type)
    if dtype.is_signed:
        raise TypeError("Don't support signed constant integer")
    int_type = IntegerType.get_signless(dtype.bit_width, context=ctx)
    return std_dialect.ConstantOp(int_type, IntegerAttr.get(int_type, node.constant_data)).result


def apply_lut(node, preds, ir_to_mlir_node, ctx):
    """Convert an arbitrary function intermediate node."""
    custom_assert(len(node.inputs) == 1, "LUT should have a single input")
    custom_assert(len(node.outputs) == 1, "LUT should have a single output")
    if not value_is_encrypted_scalar_unsigned_integer(node.inputs[0]):
        raise TypeError("Only support LUT with encrypted unsigned integers inputs")
    if not value_is_encrypted_scalar_unsigned_integer(node.outputs[0]):
        raise TypeError("Only support LUT with encrypted unsigned integers outputs")

    x_node = preds[0]
    x = ir_to_mlir_node[x_node]
    table = node.get_table()
    out_dtype = cast(Integer, node.outputs[0].data_type)
    # Create table
    dense_elem = DenseElementsAttr.get(np.array(table, dtype=np.uint64), context=ctx)
    tensor_lut = std_dialect.ConstantOp(
        RankedTensorType.get([len(table)], IntegerType.get_signless(64, context=ctx)),
        dense_elem,
    ).result
    return hlfhe.ApplyLookupTableEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, out_dtype.bit_width),
        x,
        tensor_lut,
    ).result


def dot(node, preds, ir_to_mlir_node, ctx):
    """Convert a dot intermediate node."""
    custom_assert(len(node.inputs) == 2, "Dot should have two inputs")
    custom_assert(len(node.outputs) == 1, "Dot should have a single output")
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
            f"Don't support dot between {type(node.inputs[0])} and {type(node.inputs[1])}"
        )
    lhs_node, rhs_node = preds
    # need to flip as underlying operation need encrypted first
    if value_is_clear_tensor_integer(node.inputs[0]):
        lhs_node, rhs_node = rhs_node, lhs_node
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.Dot(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


V0_OPSET_CONVERSION_FUNCTIONS = {
    ir.Add: add,
    ir.Sub: sub,
    ir.Mul: mul,
    ir.Constant: constant,
    ir.ArbitraryFunction: apply_lut,
    ir.Dot: dot,
}

# pylint: enable=no-name-in-module,no-member
