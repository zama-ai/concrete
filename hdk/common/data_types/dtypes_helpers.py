"""File to hold helper functions for data types related stuff"""

from typing import cast

from .base import BaseDataType
from .integers import Integer
from .values import BaseValue, ClearValue, EncryptedValue

INTEGER_TYPES = set([Integer])


def value_is_encrypted_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer
    """
    return (
        isinstance(value_to_check, EncryptedValue)
        and type(value_to_check.data_type) in INTEGER_TYPES
    )


def value_is_encrypted_unsigned_integer(value_to_check: BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer

    Args:
        value_to_check (BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer
    """

    return (
        value_is_encrypted_integer(value_to_check)
        and not cast(Integer, value_to_check.data_type).is_signed
    )


def find_type_to_hold_both_lossy(
    dtype1: BaseDataType,
    dtype2: BaseDataType,
) -> BaseDataType:
    """Determine the type that can represent both dtype1 and dtype2 separately, this is lossy with
        floating point types

    Args:
        dtype1 (BaseDataType): first dtype to hold
        dtype2 (BaseDataType): second dtype to hold

    Raises:
        NotImplementedError: Raised if one of the two input dtypes is not an Integer as they are the
            only type supported for now

    Returns:
        BaseDataType: The dtype able to hold (potentially lossy) dtype1 and dtype2
    """
    if isinstance(dtype1, Integer) and isinstance(dtype2, Integer):
        d1_signed = dtype1.is_signed
        d2_signed = dtype2.is_signed
        max_bits = max(dtype1.bit_width, dtype2.bit_width)

        holding_integer: BaseDataType

        if d1_signed and d2_signed:
            holding_integer = Integer(max_bits, is_signed=True)
        elif not d1_signed and not d2_signed:
            holding_integer = Integer(max_bits, is_signed=False)
        elif d1_signed and not d2_signed:
            # 2 is unsigned, if it has the bigger bit_width, we need a signed integer that can hold
            # it, so add 1 bit of sign to its bit_width
            if dtype2.bit_width >= dtype1.bit_width:
                new_bit_width = dtype2.bit_width + 1
                holding_integer = Integer(new_bit_width, is_signed=True)
            else:
                holding_integer = Integer(dtype1.bit_width, is_signed=True)
        elif not d1_signed and d2_signed:
            # Same as above, with 1 and 2 switched around
            if dtype1.bit_width >= dtype2.bit_width:
                new_bit_width = dtype1.bit_width + 1
                holding_integer = Integer(new_bit_width, is_signed=True)
            else:
                holding_integer = Integer(dtype2.bit_width, is_signed=True)

        return holding_integer

    raise NotImplementedError("For now only Integers are supported by find_type_to_hold_both_lossy")


def mix_values_determine_holding_dtype(value1: BaseValue, value2: BaseValue) -> BaseValue:
    """Returns a Value that would result from computation on both value1 and value2 while
        determining the data type able to hold both value1 and value2 data type (this can be lossy
        with floats)

    Args:
        value1 (BaseValue): first value to mix
        value2 (BaseValue): second value to mix

    Returns:
        BaseValue: The resulting mixed value with data type able to hold both value1 and value2
            dtypes
    """

    holding_type = find_type_to_hold_both_lossy(value1.data_type, value2.data_type)

    mixed_value: BaseValue

    if isinstance(value1, EncryptedValue) or isinstance(value2, EncryptedValue):
        mixed_value = EncryptedValue(holding_type)
    else:
        mixed_value = ClearValue(holding_type)

    return mixed_value
