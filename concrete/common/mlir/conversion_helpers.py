"""Helpers for MLIR conversion functionality."""

# pylint cannot extract symbol information of 'mlir' module so we need to disable some lints

# pylint: disable=no-name-in-module

from typing import Optional

from concrete.lang.dialects.fhe import EncryptedIntegerType
from mlir.ir import Context, IntegerType, RankedTensorType, Type

from ..data_types import Integer
from ..values import BaseValue, TensorValue

# pylint: enable=no-name-in-module


def integer_to_mlir_type(ctx: Context, integer: Integer, is_encrypted: bool) -> Optional[Type]:
    """Convert an integer to its corresponding MLIR type.

    Args:
        ctx (Context): the MLIR context to perform the conversion
        integer (Integer): the integer to convert
        is_encrypted (bool): whether the integer is encrypted or not

    Returns:
        Type:
            the MLIR type corresponding to given integer and encryption status
            if it's supported otherwise None
    """

    bit_width = integer.bit_width

    if is_encrypted:
        result = EncryptedIntegerType.get(ctx, bit_width)
    else:
        result = IntegerType.get_signless(bit_width)

    return result


def value_to_mlir_type(ctx: Context, value: BaseValue) -> Type:
    """Convert a value to its corresponding MLIR type.

    Args:
        ctx (Context): the MLIR context to perform the conversion
        value (BaseValue): the value to convert

    Returns:
        Type: the MLIR type corresponding to given value
    """

    dtype = value.dtype
    if isinstance(dtype, Integer):
        try:
            mlir_type = integer_to_mlir_type(ctx, dtype, value.is_encrypted)
            if isinstance(value, TensorValue):
                if not value.is_scalar:
                    mlir_type = RankedTensorType.get(value.shape, mlir_type)
                return mlir_type
        except ValueError:
            pass  # the error below will be raised

    raise TypeError(f"{value} is not supported for MLIR conversion")
