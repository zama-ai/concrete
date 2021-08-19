"""File to hold helper functions for data types related stuff."""

from copy import deepcopy
from typing import cast

from .base import BaseDataType
from .floats import Float
from .integers import Integer
from .values import BaseValue, ClearValue, EncryptedValue, ScalarValue

INTEGER_TYPES = (Integer,)
FLOAT_TYPES = (Float,)
SUPPORTED_TYPES = INTEGER_TYPES + FLOAT_TYPES


def value_is_encrypted_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer
    """
    return (
        isinstance(value_to_check, BaseValue)
        and value_to_check.is_encrypted
        and isinstance(value_to_check.data_type, INTEGER_TYPES)
    )


def value_is_encrypted_unsigned_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer and unsigned
    """
    return (
        value_is_encrypted_integer(value_to_check)
        and not cast(Integer, value_to_check.data_type).is_signed
    )


def value_is_clear_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is a clear integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a clear value of type Integer
    """
    return (
        isinstance(value_to_check, BaseValue)
        and value_to_check.is_clear
        and isinstance(value_to_check.data_type, INTEGER_TYPES)
    )


def value_is_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is of type integer.

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is a value of type Integer
    """
    return isinstance(value_to_check, BaseValue) and isinstance(
        value_to_check.data_type, INTEGER_TYPES
    )


def find_type_to_hold_both_lossy(
    dtype1: BaseDataType,
    dtype2: BaseDataType,
) -> BaseDataType:
    """Determine the type that can represent both dtype1 and dtype2 separately.

    This is lossy with floating point types.

    Args:
        dtype1 (BaseDataType): first dtype to hold
        dtype2 (BaseDataType): second dtype to hold

    Raises:
        NotImplementedError: Raised if one of the two input dtypes is not an Integer as they are the
            only type supported for now

    Returns:
        BaseDataType: The dtype able to hold (potentially lossy) dtype1 and dtype2
    """
    assert isinstance(dtype1, SUPPORTED_TYPES), f"Unsupported dtype1: {type(dtype1)}"
    assert isinstance(dtype2, SUPPORTED_TYPES), f"Unsupported dtype2: {type(dtype2)}"

    type_to_return: BaseDataType

    if isinstance(dtype1, Integer) and isinstance(dtype2, Integer):
        d1_signed = dtype1.is_signed
        d2_signed = dtype2.is_signed
        max_bits = max(dtype1.bit_width, dtype2.bit_width)

        if d1_signed and d2_signed:
            type_to_return = Integer(max_bits, is_signed=True)
        elif not d1_signed and not d2_signed:
            type_to_return = Integer(max_bits, is_signed=False)
        elif d1_signed and not d2_signed:
            # 2 is unsigned, if it has the bigger bit_width, we need a signed integer that can hold
            # it, so add 1 bit of sign to its bit_width
            if dtype2.bit_width >= dtype1.bit_width:
                new_bit_width = dtype2.bit_width + 1
                type_to_return = Integer(new_bit_width, is_signed=True)
            else:
                type_to_return = Integer(dtype1.bit_width, is_signed=True)
        elif not d1_signed and d2_signed:
            # Same as above, with 1 and 2 switched around
            if dtype1.bit_width >= dtype2.bit_width:
                new_bit_width = dtype1.bit_width + 1
                type_to_return = Integer(new_bit_width, is_signed=True)
            else:
                type_to_return = Integer(dtype2.bit_width, is_signed=True)
    elif isinstance(dtype1, Float) and isinstance(dtype2, Float):
        max_bits = max(dtype1.bit_width, dtype2.bit_width)
        type_to_return = Float(max_bits)
    elif isinstance(dtype1, Float):
        type_to_return = deepcopy(dtype1)
    elif isinstance(dtype2, Float):
        type_to_return = deepcopy(dtype2)

    return type_to_return


def mix_scalar_values_determine_holding_dtype(value1: BaseValue, value2: BaseValue) -> ScalarValue:
    """Return mixed value with data type able to hold both value1 and value2 dtypes.

    Returns a ScalarValue that would result from computation on both value1 and value2 while
    determining the data type able to hold both value1 and value2 data type (this can be lossy
    with floats).

    Args:
        value1 (BaseValue): first ScalarValue to mix.
        value2 (BaseValue): second ScalarValue to mix.

    Returns:
        ScalarValue: The resulting mixed BaseValue with data type able to hold both value1 and
            value2 dtypes.
    """

    assert isinstance(value1, ScalarValue), f"Unsupported value1: {value1}, expected ScalarValue"
    assert isinstance(value2, ScalarValue), f"Unsupported value2: {value2}, expected ScalarValue"

    holding_type = find_type_to_hold_both_lossy(value1.data_type, value2.data_type)
    mixed_value: ScalarValue

    if value1.is_encrypted or value2.is_encrypted:
        mixed_value = EncryptedValue(holding_type)
    else:
        mixed_value = ClearValue(holding_type)

    return mixed_value
