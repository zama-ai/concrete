"""
Declaration of `ClearScalar` and `EncryptedScalar` wrappers.
"""

from ..dtypes import BaseDataType
from .value import Value


def clear_scalar_builder(dtype: BaseDataType) -> Value:
    """
    Build a clear scalar value.

    Args:
        dtype (BaseDataType):
            dtype of the value

    Returns:
        Value:
            clear scalar value with given dtype
    """

    return Value(dtype=dtype, shape=(), is_encrypted=False)


ClearScalar = clear_scalar_builder


def encrypted_scalar_builder(dtype: BaseDataType) -> Value:
    """
    Build an encrypted scalar value.

    Args:
        dtype (BaseDataType):
            dtype of the value

    Returns:
        Value:
            encrypted scalar value with given dtype
    """

    return Value(dtype=dtype, shape=(), is_encrypted=True)


EncryptedScalar = encrypted_scalar_builder
