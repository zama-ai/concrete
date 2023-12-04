"""
Declaration of `ClearScalar` and `EncryptedScalar` wrappers.
"""

from ..dtypes import BaseDataType
from .value_description import ValueDescription


def clear_scalar_builder(dtype: BaseDataType) -> ValueDescription:
    """
    Build a clear scalar value.

    Args:
        dtype (BaseDataType):
            dtype of the value

    Returns:
        ValueDescription:
            clear scalar value description with given dtype
    """

    return ValueDescription(dtype=dtype, shape=(), is_encrypted=False)


ClearScalar = clear_scalar_builder


def encrypted_scalar_builder(dtype: BaseDataType) -> ValueDescription:
    """
    Build an encrypted scalar value.

    Args:
        dtype (BaseDataType):
            dtype of the value

    Returns:
        ValueDescription:
            encrypted scalar value description with given dtype
    """

    return ValueDescription(dtype=dtype, shape=(), is_encrypted=True)


EncryptedScalar = encrypted_scalar_builder
