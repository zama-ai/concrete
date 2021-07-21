"""File to hold helper functions for data types related stuff"""

from typing import cast

from . import integers, values

INTEGER_TYPES = set([integers.Integer])


def value_is_encrypted_integer(value_to_check: values.BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer

    Args:
        value_to_check (values.BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer
    """
    return (
        isinstance(value_to_check, values.EncryptedValue)
        and type(value_to_check.data_type) in INTEGER_TYPES
    )


def value_is_encrypted_unsigned_integer(value_to_check: values.BaseValue) -> bool:
    """Helper function to check that a value is an encrypted_integer

    Args:
        value_to_check (values.BaseValue): The value to check

    Returns:
        bool: True if the passed value_to_check is an encrypted value of type Integer
    """

    return (
        value_is_encrypted_integer(value_to_check)
        and not cast(integers.Integer, value_to_check.data_type).is_signed
    )
