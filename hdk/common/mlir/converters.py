"""Converter functions from HDKIR to MLIR.

Converter functions all have the same signature `converter(node, preds, ir_to_mlir_node, ctx)`
- `node`: IntermediateNode to be converted
- `preds`: List of predecessors of `node` ordered as operands
- `ir_to_mlir_node`: Dict mapping intermediate nodes to MLIR nodes or values
- `ctx`: MLIR context
"""
# pylint: disable=no-name-in-module,no-member
from typing import cast

from mlir.dialects import std as std_dialect
from mlir.ir import IntegerAttr, IntegerType
from zamalang.dialects import hlfhe

from ...common.data_types.integers import Integer
from ..data_types.dtypes_helpers import (
    value_is_clear_integer,
    value_is_encrypted_unsigned_integer,
)
from ..representation import intermediate as ir


def add(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the addition intermediate node."""
    assert len(node.inputs) == 2, "addition should have two inputs"
    assert len(node.outputs) == 1, "addition should have a single output"
    if value_is_encrypted_unsigned_integer(node.inputs[0]) and value_is_clear_integer(
        node.inputs[1]
    ):
        return _add_eint_int(node, preds, ir_to_mlir_node, ctx)
    if value_is_encrypted_unsigned_integer(node.inputs[1]) and value_is_clear_integer(
        node.inputs[0]
    ):
        # flip lhs and rhs
        return _add_eint_int(node, preds[::-1], ir_to_mlir_node, ctx)
    if value_is_encrypted_unsigned_integer(node.inputs[0]) and value_is_encrypted_unsigned_integer(
        node.inputs[1]
    ):
        return _add_eint_eint(node, preds, ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support addition between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _add_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the addition intermediate node with operands (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def _add_eint_eint(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the addition intermediate node with operands (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.AddEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def sub(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the subtraction intermediate node."""
    assert len(node.inputs) == 2, "subtraction should have two inputs"
    assert len(node.outputs) == 1, "subtraction should have a single output"
    if value_is_clear_integer(node.inputs[0]) and value_is_encrypted_unsigned_integer(
        node.inputs[1]
    ):
        return _sub_int_eint(node, preds, ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support subtraction between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _sub_int_eint(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the subtraction intermediate node with operands (int, eint)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.SubIntEintOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def mul(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the multiplication intermediate node."""
    assert len(node.inputs) == 2, "multiplication should have two inputs"
    assert len(node.outputs) == 1, "multiplication should have a single output"
    if value_is_encrypted_unsigned_integer(node.inputs[0]) and value_is_clear_integer(
        node.inputs[1]
    ):
        return _mul_eint_int(node, preds, ir_to_mlir_node, ctx)
    if value_is_encrypted_unsigned_integer(node.inputs[1]) and value_is_clear_integer(
        node.inputs[0]
    ):
        # flip lhs and rhs
        return _mul_eint_int(node, preds[::-1], ir_to_mlir_node, ctx)
    raise TypeError(
        f"Don't support multiplication between {type(node.inputs[0])} and {type(node.inputs[1])}"
    )


def _mul_eint_int(node, preds, ir_to_mlir_node, ctx):
    """Converter function for the multiplication intermediate node with operands (eint, int)."""
    lhs_node, rhs_node = preds
    lhs, rhs = ir_to_mlir_node[lhs_node], ir_to_mlir_node[rhs_node]
    return hlfhe.MulEintIntOp(
        hlfhe.EncryptedIntegerType.get(ctx, node.outputs[0].data_type.bit_width),
        lhs,
        rhs,
    ).result


def constant(node, _, __, ctx):
    """Converter function for constant inputs."""
    if not value_is_clear_integer(node.outputs[0]):
        raise TypeError("Don't support non-integer constants")
    dtype = cast(Integer, node.outputs[0].data_type)
    if dtype.is_signed:
        raise TypeError("Don't support signed constant integer")
    int_type = IntegerType.get_signless(dtype.bit_width, context=ctx)
    return std_dialect.ConstantOp(int_type, IntegerAttr.get(int_type, node.constant_data)).result


V0_OPSET_CONVERSION_FUNCTIONS = {
    ir.Add: add,
    ir.Sub: sub,
    ir.Mul: mul,
    ir.ConstantInput: constant,
}

# pylint: enable=no-name-in-module,no-member
